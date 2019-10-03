"""
Κώδικας για την ιστοσελίδα της εφαρμογής
"""

import io
import os
import os.path
import shutil
import re
import datetime
import werkzeug.utils
import werkzeug.datastructures
import werkzeug.security
import flask
import flask_sqlalchemy
import flask_login
import numpy
import cv2
import names
import stitcher
import filters
import interface

# Δημιουργία αντικειμένου app τύπου Flask καθώς και δημιουργία-αρχικοποίηση παραμέτρων του
app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config["SECRET_KEY"] = os.urandom(24)
app.config["USERS_DIR"] = "data/users"
# MAX_MBYTES_UPLOAD = 10
# app.config["MAX_CONTENT_LENGTH"] = MAX_MBYTES_UPLOAD * 1024 * 1024
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data/database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Δημιουργία αντικειμένου για τη διαχειρηση της βάσης δεδομένων για την Flask εφαρμογή
db = flask_sqlalchemy.SQLAlchemy(app)

login_manager = flask_login.LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Χρήση αντικειμένου st από την κλάση Stitcher
st = stitcher.Stitcher()

# Επιτρεπτοί τύποι αρχείων που επεξεργάζεται το πρόγραμμα
ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg")


def allowed_file(filename):
    return '.' in filename and interface.get_file_format(filename) in ALLOWED_EXTENSIONS


class User(flask_login.UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True)
    password = db.Column(db.String)    # nullable=False
    name = db.Column(db.String, default="-")
    surname = db.Column(db.String, default="-")
    email = db.Column(db.String, default="-")
    path = db.Column(db.String)
    image = db.Column(db.String, default="blank_profile.png", nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)

    images = db.relationship("Image", backref="user")
    stitched_images = db.relationship("StitchedImage", backref="user")
    # user_reactions = db.relationship("UserReaction", backref="user")

    # Κρυπτογράφηση τρέχον κωδικού
    def set_password(self, password):
        self.password = werkzeug.security.generate_password_hash(password)

    # Σύγκριση κωδικού εισόδου με τον τρέχουν
    def check_password(self, password):
        return werkzeug.security.check_password_hash(self.password, password)


# Πίνακας που προκύπτει από τη σχέση των πινάκων Image και StitchedImage
source_images = db.Table("source_images",
    db.Column("id", db.Integer, primary_key=True),
    db.Column("stitched_image_id", db.Integer, db.ForeignKey("stitched_image.image_id")),
    db.Column("image_id", db.Integer, db.ForeignKey("image.id")))


class Image(db.Model):
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    format = db.Column(db.String(4))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), primary_key=True)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    stitched_images = db.relationship("StitchedImage", backref="image", uselist=False)
    contains = db.relationship("StitchedImage", secondary=source_images, backref=db.backref("contains", lazy="dynamic"))
    # user_reactions = db.relationship("UserReaction", backref="image")


class StitchedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image.id"))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

# class UserReaction(db.Model):
#     user_id = db.Column(db.Integer, db.ForeignKey("user.id"), primary_key=True)
#     image_id = db.Column(db.Integer, db.ForeignKey("image.id"), primary_key=True)
#     reaction_type = db.Column(db.String, default="like")
#     date = db.Column(db.DateTime, default=datetime.datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.query(User).filter(User.id == user_id).first()


def get_all_images_np():
    return db.session.query(Image)\
        .filter(Image.user_id == flask_login.current_user.id)\
        .order_by(db.desc(Image.date))\
        .all()


def get_all_images(per_page=4, page=1):
    return db.session.query(Image)\
        .filter(Image.user_id == flask_login.current_user.id)\
        .order_by(db.desc(Image.date))\
        .paginate(per_page=per_page, page=page, error_out=False)


# def get_stitched_images(page=1):
#     db.session.query(Image)\
#         .join(StitchedImage)\
#         .filter(Image.user_id == flask_login.current_user.id)\
#         .order_by(db.desc(Image.date))\
#         .paginate(per_page=8, page=page, error_out=False)


# def get_images(page=1):
#     return db.session.query(Image)\
#         .filter(db.and_(Image.user_id == flask_login.current_user.id),
#                         Image.id.notin_(db.session.query(StitchedImage.image_id)))\
#         .paginate(per_page=8, page=page, error_out=False)


def get_recent_stitched_images():
    return db.session.query(Image)\
        .join(StitchedImage)\
        .filter(Image.user_id == flask_login.current_user.id)\
        .order_by(db.desc(Image.date))\
        .limit(5)\
        .all()


@app.route('/', methods=["GET"])
@app.route('/index', methods=["GET"])
@app.route('/home', methods=["GET"])
def index():
    """
    Αρχική σελίδα
    """
    data = dict()
    data["user"] = flask_login.current_user
    data["recent_stitched_images"] = dict()
    if not data["user"].is_anonymous:
        data["recent_stitched_images"] = get_recent_stitched_images()
    return flask.render_template("index.html", data=data)


def register_image(filename: str, img: numpy.ndarray, user: User, commit=False):
    """
    Έγγραφή εικόνας στη βάση δεδομένων και στο σύστημα αρχείων του χρήστη
    """

    try:
        dst_filepath = os.path.join(user.path, "library", filename)
        cv2.imwrite(dst_filepath, img)

        image = Image()
        image.id = interface.sha256sum(dst_filepath)

        filename = interface.path_leaf(dst_filepath)
        image.name = interface.get_file_name(filename)
        image.format = interface.get_file_format(filename)
        
        image.user_id = user.id

        new_filename = image.id+"."+image.format
        new_dst_filepath = os.path.join(user.path, "library", new_filename)
        try:
            os.rename(dst_filepath, new_dst_filepath)
        except FileExistsError:
            os.remove(dst_filepath)
            interface.my_print("file: '{}' already exists and has been removed!".format(filename), 1)
            return None

        db.session.add(image)
    except sqlalchemy.exc.IntegrityError:
        return None
    return image


# def like_image(image: Image, user: User, commit=False):
#     """
#     Διαχείρηση αντίδρασης χρήστη σε εικόνα.
#     """

#     ur = UserReaction(user=user, image=image, reaction_type="like")
#     try:
#         db.session.add(ur)
#         if commit:
#             db.session.commit()
#     except sqlalchemy.exc.IntegrityError:
#         return None
#     return ur


def delete_image(image, user, from_dir):
    """
    Διαχείρηση διαγραφής εικόνας χρήστη.
    """

    try:
        # remove file from database
        # db.session.query(UserReaction)\
        #     .filter(db.and_(UserReaction.image_id == image.id,
        #                     UserReaction.user_id == user.id))\
        #     .delete()
        db.session.query(StitchedImage)\
            .filter(db.and_(StitchedImage.image_id == image.id,
                            StitchedImage.user_id == user.id))\
            .delete()
        db.session.query(Image)\
            .filter(db.and_(Image.id == image.id,
                            Image.user_id == user.id))\
            .delete()
        db.session.commit()

        # remove user's files from server
        filename = image.id + "." + image.format
        os.remove(os.path.join(user.path, from_dir, filename))
    except:
        return False
    return True


@app.route('/library', methods=["GET", "POST"])
@flask_login.login_required
def library():
    """
    GET: Επιστρέφει όλες τις εικόνες της βιβλιοθήκης του χρήστη.\n
    POST: request.args:method["upload", "like", "delete"] και αιτήματα για ανέβασμα αρχείων.
    """

    if flask.request.method == "POST":
        action = flask.request.args.get("action")
        if action == "upload":
            # Ανακτηση των αρχείων από την φόρμα που συμπλήρωσε ο χρήστης (html->form->input: multiple)
            files = flask.request.files.getlist("file")

            # Ανάκτηση επιλογών φίλτρων
            autocrop = flask.request.args.get("autocrop")
            eqhist = flask.request.args.get("eqhist")
            fixaspratio = flask.request.args.get("fixaspratio")

            for file in files:
                # Έλεγχος εαν ο χρήστης δεν επίλεξε αρχεία
                if file.filename == '':
                    interface.my_print('No selected file', 2)
                    return flask.redirect(flask.request.url)
                elif file and allowed_file(file.filename):
                    file.filename = werkzeug.utils.secure_filename(file.filename)
                    
                    # Μετατροπή των δεδομένων σε μορφή συμβολοσειράς (string) σε πίνακα τύπου numpy
                    numpy_img = numpy.fromstring(file.read(), numpy.uint8)
                    
                    # Μετατροπή του προηγούμενου πίνακα σε εικόνα
                    opencv_image = cv2.imdecode(numpy_img, cv2.IMREAD_COLOR)

                    # Εφαρμογή φίλτρων (όποιων επιλέχθηκαν)
                    if autocrop == "true":
                        opencv_image = filters.automatic_cropping(opencv_image)
                    if eqhist == "true":
                        opencv_image = filters.equalize_histograms(opencv_image)
                    # if fixaspratio == "true":
                    #     opencv_image = filters.fix_aspect_ratio(opencv_image)
                    
                    # Αποθήκευση και εγγραφή εικόνας στη βάση δεδομένων
                    image = register_image(file.filename, opencv_image, flask_login.current_user)
            
            # Εφαρμογή αλλαγών στη βάση δεδομένων
            db.session.commit()
        else:
            # Ανάκτηση ονομάτων εικόνων
            images_fnames = flask.request.args.get("images").split(",")
            images = tuple()
            for fname in images_fnames:
                image = db.session.query(Image)\
                    .filter(db.and_(Image.user_id == flask_login.current_user.id,
                                    Image.id == interface.get_file_name(fname)))\
                    .first()
                images += (image,)
            
            # Αναλόγως με το αίτημα εκτελείται η κατάλληλη λειτουργία
            if "delete" in action:
                for image in images:
                    delete_image(image, flask_login.current_user, "library")
            # elif "like" in action:
            #     for image in images:
            #         like_image(image, flask_login.current_user)
            #     db.session.commit()
        return flask.redirect(flask.url_for("library"))
    else:
        # Επιστροφή πληροφοριών στη σελίδα
        data = dict()
        data["user"] = flask_login.current_user
        page = flask.request.args.get("page")
        if not page:
            page = 1
        data["all_images"] = get_all_images(per_page=8, page=int(page))
        data["ALLOWED_EXTENSIONS"] = ALLOWED_EXTENSIONS
        return flask.render_template("library.html", data=data)


@app.route("/stitch", methods=["GET", "POST"])
@flask_login.login_required
def stitch():
    """
    GET: Returns all user's images.\n
    POST: request.args: method->[1,2] and images->["1.png","2.png"].
    """

    if flask.request.method == "POST":
        method = int(flask.request.args.get("method"))
        images = flask.request.args.get("images").split(",")
        if method in stitcher.ALLOWED_METHODS and images:
            # Ανάκτηση πλήρους διαδρομής εικόνων του χρήστη
            dirs = tuple()
            for filename in images:
                dirs += (os.path.join(flask_login.current_user.path, "library", filename),)
            
            # Εφαρμογή μεθόδου συρραφής αιτούμενων εικόνων βάση της μεθόδου που επιλέχθηκε από τον χρήστη
            results = st.run(dirs, method)

            # Εαν δεν υπάρχουν αποτελέσματα τότε επέστρεψε τη σελίδα της βιβλιοθήκης
            if len(results) == 0:
                return flask.redirect(flask.url_for("library"))
            
            # cv2.imshow("test", results[0][0])
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Για κάθε αποτέλεσμα από την προηγούμενη μέθοδο
            # γίνονται οι κατάλληλες εισαγωγές στη βάση δεδομένων
            for item in results:
                new_img, src_imgs_filenames = item[0], item[1]

                # Εγγραφή και αποθήκευση της νέας εικόνας στη βάση
                image = register_image("new_img.png", new_img, flask_login.current_user)

                # Έλεγχος αν υπάρχει ήδη η εικόνα εξόδου
                if image is None:
                    return flask.redirect(flask.url_for("library"))

                st_image = StitchedImage()
                st_image.image_id = image.id
                st_image.user_id = image.user_id
                for fname in src_imgs_filenames:
                    src_img = db.session.query(Image)\
                        .filter(db.and_(Image.user_id == flask_login.current_user.id,
                                        Image.id == interface.get_file_name(interface.path_leaf(fname))))\
                        .first()
                    st_image.contains.append(src_img)
                db.session.add(st_image)
                db.session.commit()
        return flask.redirect(flask.url_for("library"))
    else:
        # Επιστροφή πληροφοριών στη σελίδα
        data = dict()
        data["user"] = flask_login.current_user
        page = flask.request.args.get("page")
        if not page:
            page = 1
        data["all_images"] = get_all_images(per_page=8, page=int(page))
        return flask.render_template("stitch.html", data=data)


def register_user(username, name=None, surname=None, email=None, password=None, commit=False):
    user = User()
    user.username = username
    user.name = name
    user.surname = surname
    user.email = email
    user.set_password(password)
    user.path = os.path.join(app.config["USERS_DIR"], user.username)

    try:
        db.session.add(user)
    except sqlalchemy.exc.IntegrityError:
        interface.my_print("User: '" + username + "' already exists!", 1)
        return None

    if os.path.isdir(user.path):
        shutil.rmtree(user.path)

    # Δημιουργία απαρραίτητων αρχείων χρήστη στον εξυπηρετητή
    os.mkdir(user.path)
    user_profile_path = os.path.join(user.path, "profile")
    os.mkdir(user_profile_path)
    shutil.copy("static/blank_profile.png", user_profile_path)
    os.mkdir(os.path.join(user.path, "library"))

    if commit:
        db.session.commit()

    return user


@app.route("/register", methods=["GET", "POST"])
def register():
    """
    GET: Returns register form.\n
    POST: Registers a new user, logs in and redirects him to homepage.
    """

    if flask.request.method == "POST":
        username = flask.request.form["username"].strip(" ")

        # Εάν υπάρχει ο χρήστης
        if db.session.query(User).filter(User.username == username).first():
            return flask.redirect(flask.url_for("register"))
            # return flask.jsonify({"error":"Please choose a different username."})

        # Ανάκτηση των δεδομένων της φόρμας εγγραφής χρήστη
        name = flask.request.form["name"].strip(" ").title()
        surname = flask.request.form["surname"].strip(" ").title()
        email = flask.request.form["email"].strip(" ")
        password = flask.request.form["password"]

        # Εγγραφή του νέου χρήστη στη βάση
        new_user = register_user(username, name, surname, email, password, commit=True)

        # Εαν δεν γίνει επιτυχή εγγραφή του νέου χρήστη τότε επιστρέφει τη σελίδα εγγραφής
        if new_user is None:
            return flask.render_template("register.html", data=dict())

        # Αποσύνδεση προηγούμενου χρήστη
        if flask_login.current_user.is_active:
            flask_login.logout_user()

        # Σύνδεση του νέου τρέχον χρήστη
        flask_login.login_user(new_user)

        return flask.redirect(flask.url_for("index"))
    else:
        data = dict()
        data["user"] = flask_login.current_user
        return flask.render_template("register.html", data=data)


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET: Returns login form.\n
    POST: Logs in a user.
    """

    if flask.request.method == "POST":
        username = flask.request.form["username"].strip(" ")

        # Έλεγχος προσπάθειας επανασύνδεσης του τρέχοντος χρήστη
        if not flask_login.current_user.is_anonymous and flask_login.current_user.username == username:
            interface.my_print("user: '" + username + "' is already logged in", 1)
            return flask.redirect(flask.url_for("login"))
            # return flask.jsonify({"error":"for a new login make sure to log out first!"})

        password = flask.request.form["password"].strip(" ")
        remember_me = False
        # if flask.request.form["RememberMe"]:
        #     remember_me = flask.request.form["RememberMe"]
        
        user = db.session.query(User)\
            .filter(User.username == username)\
            .first()
        
        if user and user.check_password(password):
            flask_login.login_user(user, remember=remember_me)
            return flask.redirect(flask.url_for("index"))

        interface.my_print("user: '" + username + "' could not be authenticated", 1)
        return flask.redirect(flask.url_for("login"))
        # return flask.jsonify({"error":"'{}' -> no matching user was found, try again".format(username)})
    else:
        data = dict()
        data["user"] = flask_login.current_user
        return flask.render_template("login.html", data=data)


def delete_user(user):
    if not user:
        return

    # remove user and it's content from database
    db.session.query(StitchedImage).filter(StitchedImage.user_id == user.id).delete()
    db.session.query(Image).filter(Image.user_id == user.id).delete()
    db.session.query(User).filter(User.id == user.id).delete()

    # remove user's files from server
    if os.path.isdir(user.path):
        shutil.rmtree(user.path)

    # log out the user
    if user.is_active:
        flask_login.logout_user()

    # commit db changes
    db.session.commit()


@app.route("/profile", methods=["GET", "POST"])
@flask_login.login_required
def profile():
    """
    GET: Returns user's profile.\n
    POST: Request to change user's profile.
    """

    if flask.request.method == "POST":
        action = flask.request.args.get("action")
        if action == "update":
            username = flask.request.args.get("update")
            return flask.redirect(flask.url_for("profile"))
        elif action == "delete":
            delete_user(flask_login.current_user)
            return flask.redirect(flask.url_for("index"))
    else:
        data = dict()
        data["user"] = flask_login.current_user
        return flask.render_template("profile.html", data=data)


@app.route("/logout", methods=["GET"])
@flask_login.login_required
def logout():
    """
    GET: It logs out a user.
    """
    flask_login.logout_user()
    return flask.redirect(flask.url_for("index"))


@app.route("/<string:from_dir>/<string:image_name>", methods=["GET"])
@flask_login.login_required
def get_image(from_dir, image_name):
    """
    GET: Επστρέφει στον περιηγητή την ζητούμενη εικόνα (εφ' όσον υπάρχει)
    """
    return flask.send_from_directory(os.path.join(flask_login.current_user.path, from_dir), image_name)


# Εκκαθάριση και επαναφορά βάσης δεδομένων
def clean_db():
    db.drop_all()
    db.create_all()


# Διαγραφή δεδομένων χρηστών
def remove_user_files():
    users_dir = "data/users"
    if os.path.isdir(users_dir):
        shutil.rmtree(users_dir)
    os.mkdir(users_dir)


# Εκκαθάριση δεδομένων της εφαρμογής
def reset():
    clean_db()
    remove_user_files()


# Σελίδα διαχειρηστή της εφαρμογής
@app.route("/admin", methods=["GET"])
def admin():
    action = flask.request.args.get("action")
    if action == "app":
        return flask.redirect(flask.url_for("index"))
    elif action == "reset":
        reset()
    elif action == "create_user":
        full_name = names.get_full_name()
        name = full_name.split()[0]
        surname = full_name.split()[1]
        user = register_user(username=full_name, name=name, surname=surname, password=full_name, commit=True)
        if flask_login.current_user.is_active:
            flask_login.logout_user()
        flask_login.login_user(user)
        path1 = "C:\\Users\\jimo4\\Downloads\\_input\\1.png"
        register_image(interface.path_leaf(path1), cv2.imread(path1), user)
        path2 = "C:\\Users\\jimo4\\Downloads\\_input\\2.png"
        register_image(interface.path_leaf(path2), cv2.imread(path2), user)
        db.session.commit()
    return flask.render_template("admin.html")
