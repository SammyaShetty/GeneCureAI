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
app.config[
    "MONGO_URI"
] = "mongodb+srv://sammyashetty16_db_user:Sammya29@cluster0.kqipwog.mongodb.net/genecure_db"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)

mongo = PyMongo(app)

# =======================
# HELPERS
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

            # Set session
            session.permanent = True
            session["user_id"] = str(user["_id"])
            session["email"] = user["email"]
            session["role"] = user["role"]
            session["name"] = user.get("name", "User")

            # Update last login
            mongo.db.users.update_one(
                {"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}}
            )

            flash(f'Welcome back, {user.get("name", "User")}!', "success")

            # Redirect based on role
            role = user["role"]
            if role == "doctor":
                return redirect(url_for("doctor_dashboard"))
            elif role == "patient":
                return redirect(url_for("patient_portal"))
            elif role == "researcher":
                # IMPORTANT: correct endpoint name
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

        # Validation
        if not all([name, email, password, confirm_password]):
            flash("All fields are required", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters", "danger")
            return render_template("register.html")

        # Prevent admin registration
        if email == "admin@genecure.com" or role == "admin":
            flash("Cannot register as admin", "danger")
            return render_template("register.html")

        # Check existing
        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user:
            flash("Email already registered", "danger")
            return redirect(url_for("login"))

        # Validate role
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
        # IMPORTANT: correct endpoint here too
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
    """
    Render doctor dashboard.
    You can later use these counts in doctor_dashboard.html with:
    {{ total_patients }}, {{ total_predictions }}, {{ active_alerts }}
    """
    total_patients = mongo.db.users.count_documents({"role": "patient"})

    # For now we don't have collections for predictions/alerts – keep 0
    total_predictions = 0
    active_alerts = 0

    return render_template(
        "doctor_dashboard.html",
        total_patients=total_patients,
        total_predictions=total_predictions,
        active_alerts=active_alerts,
    )


# Patients list page (patients.html)
@app.route("/patients")
@role_required(["doctor"])
def patients():
    # Only renders template – table is filled via JS using /api/doctor/patients
    return render_template("patients.html")


# API used by patients.html to load patient list
@app.route("/api/doctor/patients")
@role_required(["doctor"])
def api_doctor_patients():
    docs = mongo.db.users.find({"role": "patient"})

    patients = []
    for u in docs:
        patients.append(
            {
                "id": str(u["_id"]),  # used in View Details
                "name": u.get("name", ""),
                "age": u.get("age", ""),  # only if you store age
                "risk": u.get("risk_level", "N/A"),
            }
        )

    return jsonify({"patients": patients})


# Patient details (doctor view) – opened from patients.html View Details
@app.route("/doctor/patient_details")
@role_required(["doctor"])
def patient_details():
    patient_id = request.args.get("id")
    patient = None

    if patient_id:
        try:
            patient = mongo.db.users.find_one({"_id": ObjectId(patient_id)})
        except Exception as e:
            print("Error loading patient:", e)
            patient = None

    return render_template("patient_details.html", patient=patient)


@app.route("/doctor/reports")
@role_required(["doctor"])
def reports():
    return render_template("reports.html")


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
    user = get_current_user()
    # You can later load reports for this patient from "reports" collection
    reports = []
    return render_template("patient_portal.html", user=user, reports=reports)


@app.route("/patient/upload")
@role_required(["patient"])
def patient_upload():
    return render_template("patient_upload.html")


@app.route("/patient_reports")
@role_required(["patient"])
def patient_reports():
    user = get_current_user()
    reports = []
    return render_template("patient_reports.html", user=user, reports=reports)


# =======================
# RESEARCHER ROUTES
# =======================


@app.route("/researcher/reports")
@role_required(["researcher"])
def researcher_reports():
    # Show list of patients in dropdown if needed
    patients = list(mongo.db.users.find({"role": "patient"}))
    return render_template("res_reports.html", patients=patients)


# =======================
# ADMIN ROUTES
# =======================


@app.route("/admin_panel")
@role_required(["admin"])
def admin_panel():
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


# OPTIONAL: Admin APIs if you later update admin_panel.html JS
@app.route("/api/admin/users")
@role_required(["admin"])
def api_admin_users():
    users = []
    for u in mongo.db.users.find():
        users.append(
            {
                "id": str(u["_id"]),
                "name": u.get("name", ""),
                "email": u.get("email", ""),
                "role": u.get("role", ""),
                "status": "active" if u.get("is_active", True) else "inactive",
                "created_at": u.get("created_at").isoformat()
                if u.get("created_at")
                else "",
            }
        )
    return jsonify({"users": users})


@app.route("/api/admin/logs")
@role_required(["admin"])
def api_admin_logs():
    # If you later add activity_logs collection, you can fill this
    logs = []
    if "activity_logs" in mongo.db.list_collection_names():
        for log in (
            mongo.db.activity_logs.find().sort("created_at", -1).limit(50)
        ):
            logs.append(
                {
                    "time": log.get("time", ""),
                    "action": log.get("action", ""),
                    "user": log.get("user", ""),
                    "details": log.get("details", ""),
                }
            )
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
    return "Page not found", 404


@app.errorhandler(500)
def server_error(e):
    return "Server error", 500


if __name__ == "__main__":
    with app.app_context():
        try:
            mongo.db.command("ping")
            print("MongoDB connected successfully!")
            init_admin()
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            print("Please check your MONGO_URI in app.py")

    app.run(debug=True, host="0.0.0.0", port=5000)
