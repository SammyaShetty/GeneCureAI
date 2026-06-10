def init_admin():
    """Create default admin if not present."""
    if mongo is None:
        print("MongoDB not connected. Skipping admin initialization.")
        return
    try:
        admin_exists = mongo.db.users.find_one({"email": "admin@genecure.com"})
        if not admin_exists:
            admin_user = {
                "email": "admin@genecure.com",
                "password": generate_password_hash("admin123"),
                "role": "admin",
                "name": "System Administrator",
                "created_at": datetime.utcnow(),
                "is_active": True,
            }
            mongo.db.users.insert_one(admin_user)
            print("Admin user created successfully!")
    except Exception as e:
        print#!/usr/bin/env python3
"""
GeneCure AI - Integrated Application
Combines authentication, role-based access, and ML predictions
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
)
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
from bson.objectid import ObjectId

app = Flask(__name__)

# =======================
# CONFIG
# =======================
app.config["SECRET_KEY"] = "your-secret-key-change-this-in-production"

# MongoDB Configuration - UPDATE THIS WITH YOUR CORRECT CONNECTION STRING
# Option 1: MongoDB Atlas (Cloud)
# Get the correct connection string from MongoDB Atlas Dashboard:
# 1. Go to https://cloud.mongodb.com
# 2. Click "Connect" on your cluster
# 3. Choose "Connect your application"
# 4. Copy the connection string and replace below
MONGO_URI = ""

# Option 2: Local MongoDB (if Atlas doesn't work)
# Uncomment below and comment out the Atlas URI above
# MONGO_URI = "mongodb://localhost:27017/genecure_db"

app.config["MONGO_URI"] = MONGO_URI
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)

# Initialize MongoDB with error handling
try:
    mongo = PyMongo(app)
    print("PyMongo initialized successfully")
except Exception as e:
    print(f"MongoDB initialization error: {e}")
    print("\nPlease check your MongoDB connection string!")
    print("Visit https://cloud.mongodb.com to get the correct connection string")
    mongo = None

# =======================
# ML MODELS SETUP
# =======================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "genecure_models"

# Load trained models
try:
    model1 = joblib.load(MODELS_DIR / "model1_cancer_detection.pkl")
    label_encoder1 = joblib.load(MODELS_DIR / "model1_label_encoder.pkl")
    
    drug_ranking_path = MODELS_DIR / "model3_drug_ranking.csv"
    drug_ranking_df = pd.read_csv(drug_ranking_path) if drug_ranking_path.exists() else pd.DataFrame()
    
    print("ML models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load ML models: {e}")
    model1 = None
    label_encoder1 = None
    drug_ranking_df = pd.DataFrame()

# =======================
# ML HELPER FUNCTIONS
# =======================

def categorize_stage(stage_str: str) -> str:
    """Convert free-text stage into: 'early', 'late', or 'unknown'"""
    if not stage_str:
        return "unknown"
    s = stage_str.strip().upper()

    if "STAGE I" in s or "STAGE II" in s:
        if "III" in s or "IV" in s:
            return "late"
        return "early"
    if " III" in s or " IV" in s or "STAGE III" in s or "STAGE IV" in s:
        return "late"
    if " I" in s or " II" in s:
        return "early"

    return "unknown"


def rule_based_risk(scenario: str, tumor_stage: str):
    """
    Returns: risk_category, high_prob, low_prob
    Simple logic for demo:
      - EARLY_SCREEN + early/unknown stage -> LOW RISK
      - KNOWN_CANCER + Stage II -> INTERMEDIATE
      - KNOWN_CANCER + Stage III/IV -> HIGH RISK
    """
    scenario = (scenario or "").upper()
    stage_cat = categorize_stage(tumor_stage)

    risk_category = "HIGH RISK"
    high_prob = 0.85
    low_prob = 0.15

    if scenario == "EARLY_SCREEN":
        if stage_cat in ["early", "unknown"]:
            risk_category = "LOW RISK"
            high_prob = 0.20
            low_prob = 0.80
        else:
            risk_category = "INTERMEDIATE RISK"
            high_prob = 0.50
            low_prob = 0.50
    elif scenario == "KNOWN_CANCER":
        if stage_cat == "early":
            risk_category = "INTERMEDIATE RISK"
            high_prob = 0.55
            low_prob = 0.45
        elif stage_cat == "late":
            risk_category = "HIGH RISK"
            high_prob = 0.90
            low_prob = 0.10
        else:
            risk_category = "INTERMEDIATE RISK"
            high_prob = 0.60
            low_prob = 0.40

    return risk_category, high_prob, low_prob


def suggest_drugs(cancer_type: str, risk_category: str, stage_str: str, 
                  drug_ranking_df: pd.DataFrame, top_n: int = 5):
    """Personalized drug recommendation using cancer type, risk, and stage"""
    if drug_ranking_df is None or drug_ranking_df.empty:
        return []

    cancer_type = (cancer_type or "").upper()
    risk_category = (risk_category or "").upper()
    stage_cat = categorize_stage(stage_str)

    df = drug_ranking_df.sort_values("mean_ic50", ascending=True).copy()

    if risk_category == "HIGH RISK" or stage_cat == "late":
        start_idx, end_idx = 0, 15
    elif risk_category == "LOW RISK" and stage_cat == "early":
        start_idx, end_idx = 10, 35
    else:
        start_idx, end_idx = 5, 25

    start_idx = max(0, min(start_idx, len(df) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(df)))

    df_segment = df.iloc[start_idx:end_idx].copy()
    df_segment["drug_upper"] = df_segment["drug"].astype(str).str.upper()

    cancer_keywords = {
        "BRCA": ["OLAPAR", "FULVES", "DOCETAX", "GEMCIT", "CISPL"],
        "LUAD": ["GEMCIT", "CISPL", "SN-38", "PICTIL", "AFATIN"],
        "COAD": ["SN-38", "GEMCIT", "CISPL", "PICTIL"],
        "KIRC": ["GEMCIT", "CISPL", "AZD7762"],
        "PRAD": ["DOCETAX", "GEMCIT", "CISPL"],
    }
    keywords = cancer_keywords.get(cancer_type, [])

    if keywords:
        mask = False
        for kw in keywords:
            mask = mask | df_segment["drug_upper"].str.contains(kw, na=False)
        df_pref = df_segment[mask].copy()
    else:
        df_pref = df_segment.iloc[0:0].copy()

    results = []

    for _, row in df_pref.iterrows():
        results.append({
            "drug": row["drug"],
            "mean_ic50": float(row["mean_ic50"]),
            "category": row.get("category", "Segment-preferred"),
            "model_r2": float(row.get("model_r2", 0.0)),
            "reason": f"Within {risk_category.lower()} segment and matched cancer type",
        })
        if len(results) >= top_n:
            return results

    used = {r["drug"] for r in results}
    for _, row in df_segment.iterrows():
        if row["drug"] in used:
            continue
        results.append({
            "drug": row["drug"],
            "mean_ic50": float(row["mean_ic50"]),
            "category": row.get("category", "Segment-selected"),
            "model_r2": float(row.get("model_r2", 0.0)),
            "reason": f"From {risk_category.lower()} segment (IC50-based ranking)",
        })
        if len(results) >= top_n:
            break

    return results


def run_models_for_patient(scenario: str, tumor_stage: str):
    """Run all 3 ML models for a patient"""
    if model1 is None or label_encoder1 is None:
        return None, None, None

    # Model 1 - Cancer Type Detection
    n_pca = model1.n_features_in_

    if scenario == "KNOWN_CANCER":
        X1 = np.ones((1, n_pca), dtype=float) * 2.0
    else:
        X1 = np.zeros((1, n_pca), dtype=float)

    proba1 = model1.predict_proba(X1)[0]
    pred_idx1 = int(np.argmax(proba1))
    pred_class1 = label_encoder1.inverse_transform([pred_idx1])[0]
    confidence1 = float(proba1[pred_idx1])

    cancer_result = {
        "predicted_cancer_type": pred_class1,
        "confidence": confidence1,
    }

    # Model 2 - Rule-based Risk
    risk_category, high_prob, low_prob = rule_based_risk(scenario, tumor_stage)

    risk_result = {
        "risk_label": risk_category,
        "high_risk_prob": float(high_prob),
        "low_risk_prob": float(low_prob),
        "confidence": float(max(high_prob, low_prob)),
    }

    # Model 3 - Personalized Drugs
    top_drugs = suggest_drugs(
        cancer_type=cancer_result["predicted_cancer_type"],
        risk_category=risk_result["risk_label"],
        stage_str=tumor_stage,
        drug_ranking_df=drug_ranking_df,
        top_n=5,
    )

    drug_result = {"top_drugs": top_drugs}

    return cancer_result, risk_result, drug_result


# =======================
# AUTH HELPERS
# =======================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to access this page", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                flash("Please login to access this page", "warning")
                return redirect(url_for("login"))

            if session.get("role") not in allowed_roles:
                flash("You do not have permission to access this page", "danger")
                return redirect(url_for("dashboard"))

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def get_current_user():
    """Return the full user document for the logged-in user, or None."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        return mongo.db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


def init_admin():
    """Create default admin if not present."""
    try:
        admin_exists = mongo.db.users.find_one({"email": "admin@genecure.com"})
        if not admin_exists:
            admin_user = {
                "email": "admin@genecure.com",
                "password": generate_password_hash("admin123"),
                "role": "admin",
                "name": "System Administrator",
                "created_at": datetime.utcnow(),
                "is_active": True,
            }
            mongo.db.users.insert_one(admin_user)
            print("Admin user created successfully!")
    except Exception as e:
        print(f"Error creating admin: {e}")


# =======================
# AUTH / PUBLIC ROUTES
# =======================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please provide both email and password", "danger")
            return render_template("login.html")

        user = mongo.db.users.find_one({"email": email})

        if user and check_password_hash(user["password"], password):
            if not user.get("is_active", True):
                flash("Your account has been deactivated", "danger")
                return render_template("login.html")

            session.permanent = True
            session["user_id"] = str(user["_id"])
            session["email"] = user["email"]
            session["role"] = user["role"]
            session["name"] = user.get("name", "User")

            mongo.db.users.update_one(
                {"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}}
            )

            flash(f'Welcome back, {user.get("name", "User")}!', "success")

            role = user["role"]
            if role == "doctor":
                return redirect(url_for("doctor_dashboard"))
            elif role == "patient":
                return redirect(url_for("patient_portal"))
            elif role == "researcher":
                return redirect(url_for("researcher_reports"))
            elif role == "admin":
                return redirect(url_for("admin_panel"))
            else:
                return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", "patient")
        phone = request.form.get("phone", "").strip()

        if not all([name, email, password, confirm_password]):
            flash("All fields are required", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters", "danger")
            return render_template("register.html")

        if email == "admin@genecure.com" or role == "admin":
            flash("Cannot register as admin", "danger")
            return render_template("register.html")

        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user:
            flash("Email already registered", "danger")
            return redirect(url_for("login"))

        if role not in ["doctor", "patient", "researcher"]:
            role = "patient"

        new_user = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
            "role": role,
            "phone": phone,
            "created_at": datetime.utcnow(),
            "is_active": True,
        }

        try:
            mongo.db.users.insert_one(new_user)
            flash("Registration successful! Please login", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash("Registration failed", "danger")
            print(f"Registration error: {e}")

    return render_template("register.html")


@app.route("/dashboard")
@login_required
def dashboard():
    role = session.get("role")

    if role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    elif role == "patient":
        return redirect(url_for("patient_portal"))
    elif role == "researcher":
        return redirect(url_for("researcher_reports"))
    elif role == "admin":
        return redirect(url_for("admin_panel"))

    return render_template("dashboard.html")


# =======================
# DOCTOR ROUTES
# =======================

@app.route("/doctor/dashboard")
@role_required(["doctor"])
def doctor_dashboard():
    """Doctor dashboard with statistics"""
    total_patients = mongo.db.users.count_documents({"role": "patient"})
    total_predictions = mongo.db.predictions.count_documents({}) if "predictions" in mongo.db.list_collection_names() else 0
    active_alerts = 0

    return render_template(
        "doctor_dashboard.html",
        total_patients=total_patients,
        total_predictions=total_predictions,
        active_alerts=active_alerts,
    )


@app.route("/patients")
@role_required(["doctor"])
def patients():
    """Patient list page"""
    return render_template("patients.html")


@app.route("/api/doctor/patients")
@role_required(["doctor"])
def api_doctor_patients():
    """API endpoint for patient list"""
    docs = mongo.db.users.find({"role": "patient"})

    patients = []
    for u in docs:
        patients.append({
            "id": str(u["_id"]),
            "name": u.get("name", ""),
            "age": u.get("age", "N/A"),
            "risk": u.get("risk_level", "N/A"),
        })

    return jsonify({"patients": patients})


@app.route("/doctor/patient_details")
@role_required(["doctor"])
def patient_details():
    """Patient details view for doctors"""
    patient_id = request.args.get("id")
    patient = None
    predictions = None

    if patient_id:
        try:
            patient = mongo.db.users.find_one({"_id": ObjectId(patient_id)})
            
            # Load latest prediction for this patient
            if "predictions" in mongo.db.list_collection_names():
                predictions = mongo.db.predictions.find_one(
                    {"patient_id": patient_id},
                    sort=[("created_at", -1)]
                )
        except Exception as e:
            print("Error loading patient:", e)

    return render_template("patient_details.html", patient=patient, predictions=predictions)


@app.route("/doctor/run_prediction", methods=["POST"])
@role_required(["doctor"])
def doctor_run_prediction():
    """Run ML prediction for a patient (doctor initiated)"""
    try:
        patient_id = request.form.get("patient_id")
        scenario = request.form.get("scenario", "EARLY_SCREEN").upper()
        tumor_stage = request.form.get("tumor_stage", "Unknown")
        
        if not patient_id:
            flash("Patient ID is required", "danger")
            return redirect(url_for("patients"))
        
        # Run ML models
        cancer_result, risk_result, drug_result = run_models_for_patient(scenario, tumor_stage)
        
        if cancer_result is None:
            flash("ML models not loaded. Cannot run prediction.", "danger")
            return redirect(url_for("patient_details", id=patient_id))
        
        # Save prediction to database
        prediction_doc = {
            "patient_id": patient_id,
            "doctor_id": session.get("user_id"),
            "scenario": scenario,
            "tumor_stage": tumor_stage,
            "cancer_result": cancer_result,
            "risk_result": risk_result,
            "drug_result": drug_result,
            "created_at": datetime.utcnow(),
        }
        
        mongo.db.predictions.insert_one(prediction_doc)
        
        # Update patient risk level
        mongo.db.users.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": {"risk_level": risk_result["risk_label"]}}
        )
        
        flash("Prediction completed successfully!", "success")
        return redirect(url_for("patient_details", id=patient_id))
        
    except Exception as e:
        print(f"Prediction error: {e}")
        flash("Error running prediction", "danger")
        return redirect(url_for("patients"))


@app.route("/doctor/reports")
@role_required(["doctor"])
def reports():
    # Get all patients for the dropdown
    patients = list(mongo.db.users.find({"role": "patiet"}))
    return render_template("reports.html", patients=patients)


@app.route("/api/doctor/patient/<patient_id>")
@role_required(["doctor"])
def api_get_patient_with_prediction(patient_id):
    """API endpoint to get patient data with latest prediction"""
    try:
        # Get patient
        patient = mongo.db.users.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
        
        # Get latest prediction
        prediction = None
        if "predictions" in mongo.db.list_collection_names():
            prediction = mongo.db.predictions.find_one(
                {"patient_id": patient_id},
                sort=[("created_at", -1)]
            )
        
        # Format response
        response = {
            "patient": {
                "id": str(patient["_id"]),
                "name": patient.get("name", ""),
                "age": patient.get("age", "N/A"),
                "gender": patient.get("gender", "N/A"),
                "email": patient.get("email", ""),
            }
        }
        
        if prediction:
            response["prediction"] = {
                "cancer_result": prediction.get("cancer_result"),
                "risk_result": prediction.get("risk_result"),
                "drug_result": prediction.get("drug_result"),
                "tumor_stage": prediction.get("tumor_stage"),
                "scenario": prediction.get("scenario"),
                "created_at": prediction.get("created_at").isoformat() if prediction.get("created_at") else None
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error fetching patient data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/doctor/variant_explorer")
@role_required(["doctor"])
def variant_explorer():
    return render_template("variant_explorer.html")


# =======================
# PATIENT ROUTES
# =======================

@app.route("/patient/portal")
@role_required(["patient"])
def patient_portal():
    """Patient portal with their reports"""
    user = get_current_user()
    patient_id = str(user["_id"])
    
    # Load all predictions for this patient
    reports = []
    if "predictions" in mongo.db.list_collection_names():
        reports = list(mongo.db.predictions.find(
            {"patient_id": patient_id}
        ).sort("created_at", -1))
    
    return render_template("patient_portal.html", user=user, reports=reports)


@app.route("/patient/upload", methods=["GET", "POST"])
@role_required(["patient"])
def patient_upload():
    """Patient file upload and self-prediction"""
    if request.method == "POST":
        scenario = request.form.get("scenario", "EARLY_SCREEN").upper()
        tumor_stage = request.form.get("tumor_stage", "Unknown")
        
        user = get_current_user()
        patient_id = str(user["_id"])
        
        # Handle CSV upload
        if "patient_file" in request.files and request.files["patient_file"].filename != "":
            file = request.files["patient_file"]
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    patient_row = df.to_dict(orient="records")[0]
                    scenario = str(patient_row.get("scenario", "EARLY_SCREEN")).strip().upper()
                    tumor_stage = str(patient_row.get("tumor_stage", "Unknown"))
            except Exception as e:
                flash(f"Error reading CSV: {e}", "danger")
                return redirect(url_for("patient_upload"))
        
        # Run prediction
        cancer_result, risk_result, drug_result = run_models_for_patient(scenario, tumor_stage)
        
        if cancer_result is None:
            flash("ML models not loaded. Cannot run prediction.", "danger")
            return redirect(url_for("patient_upload"))
        
        # Save prediction
        prediction_doc = {
            "patient_id": patient_id,
            "scenario": scenario,
            "tumor_stage": tumor_stage,
            "cancer_result": cancer_result,
            "risk_result": risk_result,
            "drug_result": drug_result,
            "created_at": datetime.utcnow(),
        }
        
        mongo.db.predictions.insert_one(prediction_doc)
        
        # Update patient risk level
        mongo.db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"risk_level": risk_result["risk_label"]}}
        )
        
        flash("Analysis completed! View your results in the portal.", "success")
        return redirect(url_for("patient_portal"))
    
    return render_template("patient_upload.html")


@app.route("/patient_reports")
@role_required(["patient"])
def patient_reports():
    """Detailed patient reports page"""
    user = get_current_user()
    patient_id = str(user["_id"])
    
    reports = []
    if "predictions" in mongo.db.list_collection_names():
        reports = list(mongo.db.predictions.find(
            {"patient_id": patient_id}
        ).sort("created_at", -1))
    
    return render_template("patient_reports.html", user=user, reports=reports)


# =======================
# RESEARCHER ROUTES
# =======================

@app.route("/researcher/reports")
@role_required(["researcher"])
def researcher_reports():
    """Researcher dashboard with all patient data"""
    patients = list(mongo.db.users.find({"role": "patient"}))
    
    # Load all predictions
    all_predictions = []
    if "predictions" in mongo.db.list_collection_names():
        all_predictions = list(mongo.db.predictions.find().sort("created_at", -1).limit(100))
    
    return render_template("res_reports.html", patients=patients, predictions=all_predictions)


@app.route("/api/researcher/patient/<patient_id>")
@role_required(["researcher"])
def api_get_researcher_patient(patient_id):
    """API endpoint for researchers to get patient data with prediction"""
    try:
        # Get patient
        patient = mongo.db.users.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
        
        # Get latest prediction
        prediction = None
        if "predictions" in mongo.db.list_collection_names():
            prediction = mongo.db.predictions.find_one(
                {"patient_id": patient_id},
                sort=[("created_at", -1)]
            )
        
        # Format response
        response = {
            "patient": {
                "id": str(patient["_id"]),
                "name": patient.get("name", ""),
                "age": patient.get("age", "N/A"),
                "gender": patient.get("gender", "N/A"),
                "email": patient.get("email", ""),
            }
        }
        
        if prediction:
            response["prediction"] = {
                "cancer_result": prediction.get("cancer_result"),
                "risk_result": prediction.get("risk_result"),
                "drug_result": prediction.get("drug_result"),
                "tumor_stage": prediction.get("tumor_stage"),
                "scenario": prediction.get("scenario"),
                "created_at": prediction.get("created_at").isoformat() if prediction.get("created_at") else None
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error fetching patient data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/researcher/analytics")
@role_required(["researcher"])
def researcher_analytics():
    """Analytics dashboard for researchers"""
    # Aggregate statistics
    stats = {
        "total_predictions": 0,
        "cancer_types": {},
        "risk_levels": {},
    }
    
    if "predictions" in mongo.db.list_collection_names():
        stats["total_predictions"] = mongo.db.predictions.count_documents({})
        
        # Count by cancer type
        pipeline = [
            {"$group": {"_id": "$cancer_result.predicted_cancer_type", "count": {"$sum": 1}}}
        ]
        for doc in mongo.db.predictions.aggregate(pipeline):
            stats["cancer_types"][doc["_id"]] = doc["count"]
        
        # Count by risk level
        pipeline = [
            {"$group": {"_id": "$risk_result.risk_label", "count": {"$sum": 1}}}
        ]
        for doc in mongo.db.predictions.aggregate(pipeline):
            stats["risk_levels"][doc["_id"]] = doc["count"]
    
    return render_template("researcher_analytics.html", stats=stats)


# =======================
# ADMIN ROUTES
# =======================

@app.route("/admin_panel")
@role_required(["admin"])
def admin_panel():
    """Admin dashboard"""
    try:
        total_users = mongo.db.users.count_documents({})
        total_doctors = mongo.db.users.count_documents({"role": "doctor"})
        total_patients = mongo.db.users.count_documents({"role": "patient"})
        total_researchers = mongo.db.users.count_documents({"role": "researcher"})

        recent_users = list(mongo.db.users.find().sort("created_at", -1).limit(10))

        return render_template(
            "admin_panel.html",
            total_users=total_users,
            total_doctors=total_doctors,
            total_patients=total_patients,
            total_researchers=total_researchers,
            recent_users=recent_users,
        )
    except Exception as e:
        print("Admin panel error:", e)
        return render_template(
            "admin_panel.html",
            total_users=0,
            total_doctors=0,
            total_patients=0,
            total_researchers=0,
            recent_users=[],
        )


@app.route("/api/admin/users")
@role_required(["admin"])
def api_admin_users():
    """API for admin user management"""
    users = []
    for u in mongo.db.users.find():
        users.append({
            "id": str(u["_id"]),
            "name": u.get("name", ""),
            "email": u.get("email", ""),
            "role": u.get("role", ""),
            "status": "active" if u.get("is_active", True) else "inactive",
            "created_at": u.get("created_at").isoformat() if u.get("created_at") else "",
        })
    return jsonify({"users": users})


@app.route("/api/admin/logs")
@role_required(["admin"])
def api_admin_logs():
    """API for admin activity logs"""
    logs = []
    if "activity_logs" in mongo.db.list_collection_names():
        for log in mongo.db.activity_logs.find().sort("created_at", -1).limit(50):
            logs.append({
                "time": log.get("time", ""),
                "action": log.get("action", ""),
                "user": log.get("user", ""),
                "details": log.get("details", ""),
            })
    return jsonify({"logs": logs})


# =======================
# LOGOUT & ERRORS
# =======================

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(e):
    return "404 - Page not found", 404


@app.errorhandler(500)
def server_error(e):
    return "500 - Server error", 500


# =======================
# MAIN
# =======================

if __name__ == "__main__":
    with app.app_context():
        try:
            mongo.db.command("ping")
            print("MongoDB connected successfully!")
            init_admin()
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            print("Please check your MONGO_URI")

    app.run(debug=True, host="0.0.0.0", port=5000)
