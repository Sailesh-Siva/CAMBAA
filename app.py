from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for
)

import os
import uuid

from werkzeug.utils import secure_filename

# =========================================================
# BONE AGE INFERENCE
# =========================================================

from utils import (
    predict_bone_age_model1,
    predict_bone_age_model2
)


# =========================================================
# CONFIG
# =========================================================

UPLOAD_FOLDER = 'uploads'

RESULT_FOLDER = 'static/results'

ALLOWED_EXTENSIONS = {
    'png',
    'jpg',
    'jpeg'
}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['RESULT_FOLDER'] = RESULT_FOLDER

app.secret_key = 'your_secret_key_here'

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(filename):

    return (
        '.' in filename
        and
        filename.rsplit('.', 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# ROOT
# =========================================================

@app.route('/')
def root():

    return redirect(
        url_for('boneage')
    )


# =========================================================
# BONE AGE ROUTE
# =========================================================

@app.route(
    '/boneage',
    methods=['GET', 'POST']
)
def boneage():

    if request.method == 'POST':

        # =================================================
        # VALIDATE FILE
        # =================================================

        if 'image' not in request.files:

            return render_template(
                'boneage.html',
                error="No file uploaded."
            )

        uploaded_file = request.files['image']

        if uploaded_file.filename == '':

            return render_template(
                'boneage.html',
                error="No image selected."
            )

        if not allowed_file(uploaded_file.filename):

            return render_template(
                'boneage.html',
                error="Unsupported file format."
            )

        # =================================================
        # FORM DATA
        # =================================================

        gender = request.form.get('gender')

        age_group = request.form.get('age_group')

        model_type = request.form.get("model_type")

        # =================================================
        # SAVE INPUT IMAGE
        # =================================================

        filename = (
            f"{uuid.uuid4().hex}_"
            f"{secure_filename(uploaded_file.filename)}"
        )

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )

        uploaded_file.save(filepath)

        # =================================================
        # RESULT FILE NAME
        # =================================================

        result_filename = (
            f"result_"
            f"{uuid.uuid4().hex}.png"
        )

        # =================================================
        # GENDER TO NUMERIC
        # Male   = 1
        # Female = 0
        # =================================================

        gender_value = (
            1 if gender == "male"
            else 0
        )

        # =================================================
        # INFERENCE
        # =================================================

        try:
            
            if model_type == "model1":

                predicted_age, result_path = predict_bone_age_model1(
                    image_path=filepath,
                    gender_value=gender_value,
                    svr_type=age_group,
                    output_filename=result_filename
                )

            elif model_type == "model2":

                predicted_age, result_path = predict_bone_age_model2(
                    image_path=filepath,
                    gender_value=gender_value,
                    svr_type=age_group,
                    output_filename=result_filename
                )

            else:

                raise ValueError(
                    "Please select a prediction model."
                )

        except Exception as e:

            return render_template(
                'boneage.html',
                error=str(e)
            )

        # =================================================
        # RENDER RESULTS
        # =================================================

        return render_template(

            'boneage.html',

            uploaded_image=filename,

            result_image=result_filename,

            predicted_age=round(
                predicted_age,
                2
            ),

            gender=gender.capitalize(),

            age_group=(
                "Below 8 Years"
                if age_group == "below_8"
                else "Above 8 Years"
            )
        )

    # =====================================================
    # DEFAULT PAGE
    # =====================================================

    return render_template(

        'boneage.html',

        uploaded_image=None,

        result_image=None,

        predicted_age=None,

        gender=None,

        age_group=None,

        error=None
    )


# =========================================================
# STATIC RESULT ROUTE
# =========================================================

@app.route('/static/results/<filename>')
def result_file(filename):

    return send_from_directory(
        app.config['RESULT_FOLDER'],
        filename
    )


# =========================================================
# UPLOAD ROUTE
# =========================================================

@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


# =========================================================
# ABOUT PAGE
# =========================================================

@app.route('/about')
def about():

    return render_template(
        'about.html'
    )


# =========================================================
# METRICS PAGE
# =========================================================

@app.route('/metrics')
def metrics():

    return render_template(
        'metrics.html'
    )


# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 7860))

    app.run(host="0.0.0.0", port=port)