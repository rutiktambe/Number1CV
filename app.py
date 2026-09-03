import streamlit as st
import PyPDF2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Number1CV | Rutik Tambe",
    page_icon="📄",
    layout="centered"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');


/* ---------------------------------------------------------
   GLOBAL FONT
--------------------------------------------------------- */

html,
body,
[class*="css"],
[data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}


/* ---------------------------------------------------------
   BACKGROUND
   Let Streamlit control Light / Dark Mode automatically
--------------------------------------------------------- */

.stApp {
    background: transparent !important;
}


/* ---------------------------------------------------------
   MAIN CONTAINER
--------------------------------------------------------- */

.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}


/* ---------------------------------------------------------
   LOGO
--------------------------------------------------------- */

.main-title {
    text-align: center;
    font-size: 88px;
    font-weight: 900;

    /* Automatically follows Streamlit theme */
    color: inherit;

    letter-spacing: -5px;
    margin-bottom: 15px;
    line-height: 1;

    text-shadow:
        0 5px 15px rgba(0, 0, 0, 0.15);
}


/* ---------------------------------------------------------
   RED NUMBER 1
--------------------------------------------------------- */

.logo-one {
    display: inline-flex;
    justify-content: center;
    align-items: center;

    width: 76px;
    height: 76px;

    margin: 0 6px;

    border-radius: 50%;

    background: linear-gradient(
        145deg,
        #ff3b3b,
        #e50914,
        #a90000
    );

    color: white !important;

    font-size: 48px;
    font-weight: 900;

    vertical-align: middle;

    box-shadow:
        0 8px 20px rgba(229, 9, 20, 0.35),
        inset 0 3px 5px rgba(255, 255, 255, 0.25);
}


/* ---------------------------------------------------------
   TAGLINE
--------------------------------------------------------- */

.subtitle {
    text-align: center;
    font-size: 17px;

    color: inherit;

    opacity: 0.65;

    margin-bottom: 30px;
}


/* ---------------------------------------------------------
   HEADINGS
--------------------------------------------------------- */

h1,
h2,
h3 {
    font-weight: 800 !important;
}


/* ---------------------------------------------------------
   TEXT INPUT
--------------------------------------------------------- */

[data-testid="stTextInput"] input {
    border-radius: 12px !important;
}


/* ---------------------------------------------------------
   TEXT AREA
--------------------------------------------------------- */

[data-testid="stTextArea"] textarea {
    border-radius: 12px !important;
}


/* ---------------------------------------------------------
   FILE UPLOADER
--------------------------------------------------------- */

[data-testid="stFileUploader"] {
    padding: 18px;

    border-radius: 16px;
}


/* ---------------------------------------------------------
   UPLOAD BUTTON
--------------------------------------------------------- */

[data-testid="stFileUploader"] button {
    border-radius: 10px !important;

    font-weight: 600 !important;
}


/* ---------------------------------------------------------
   ANALYZE BUTTON
--------------------------------------------------------- */

.stButton > button {
    height: 54px;

    border-radius: 14px;

    border: none;

    font-size: 16px;

    font-weight: 700;

    color: white !important;

    background:
        linear-gradient(
            135deg,
            #111111,
            #292929
        );

    box-shadow:
        0 8px 18px rgba(0, 0, 0, 0.18);

    transition:
        all 0.2s ease;
}


.stButton > button:hover {
    background:
        linear-gradient(
            135deg,
            #e50914,
            #b30000
        );

    transform:
        translateY(-1px);
}


/* ---------------------------------------------------------
   METRIC
--------------------------------------------------------- */

[data-testid="stMetric"] {
    padding: 20px;

    border-radius: 16px;
}


/* ---------------------------------------------------------
   PROGRESS BAR
--------------------------------------------------------- */

.stProgress > div > div {
    background-color: #e50914 !important;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {
    text-align: center;

    padding: 25px 10px 15px;

    width: 100%;
}


.footer-name {
    font-size: 17px;

    font-weight: 700;

    color: inherit;

    margin-bottom: 8px;
}


.footer-name span {
    color: #e50914 !important;
}


.footer-skills {
    font-size: 14px;

    color: inherit;

    opacity: 0.65;

    margin-bottom: 8px;
}


.footer-tech {
    font-size: 12px;

    color: inherit;

    opacity: 0.45;
}


/* ---------------------------------------------------------
   MOBILE RESPONSIVE
--------------------------------------------------------- */

@media (max-width: 768px) {

    .block-container {
        padding-top: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }


    .main-title {
        font-size: 58px;

        letter-spacing: -3px;
    }


    .logo-one {
        width: 54px;

        height: 54px;

        font-size: 34px;
    }


    .subtitle {
        font-size: 14px;
    }

}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text_from_pdf(uploaded_file):

    try:

        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception:
        return ""


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        Number<span class="logo-one">1</span>CV
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
        Our most intelligent Resume Matching using NLP & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ---------------------------------------------------------
# INTRODUCTION
# ---------------------------------------------------------

st.info(
    "Upload a candidate's resume and enter the job requirements. "
    "Number1CV will analyze the resume and calculate the candidate's match score."
)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

st.subheader("Upload Candidate Resume")


uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)


# ---------------------------------------------------------
# JOB DESCRIPTION
# ---------------------------------------------------------

st.subheader("Job Description")


job_desc = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder="""For Example: We are looking for a Data Analyst with knowledge of:

Python
SQL
Pandas
NumPy
Tensorflow
Machine Learning
Data Visualization
"""
)


# ---------------------------------------------------------
# REQUIRED SKILLS
# ---------------------------------------------------------

st.subheader("Required Skills")


skills_input = st.text_input(
    "Enter required skills separated by commas",
    placeholder="Python, SQL, Pandas, NumPy, Machine Learning"
)


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

if st.button(
    "Analyze Candidate",
    use_container_width=True
):


    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    if uploaded_file is None:

        st.warning(
            "Please upload a Resume PDF before analyzing."
        )


    elif not job_desc.strip():

        st.warning(
            "Please enter the Job Description before analyzing."
        )


    else:


        # -------------------------------------------------
        # EXTRACT RESUME TEXT
        # -------------------------------------------------

        resume_text = extract_text_from_pdf(uploaded_file)


        if not resume_text:

            st.error(
                "Unable to extract text from this PDF. "
                "Please upload a valid text-based PDF."
            )


        else:


            # -------------------------------------------------
            # RESUME MATCH SCORE
            # -------------------------------------------------

            try:

                documents = [
                    resume_text,
                    job_desc
                ]


                vectorizer = CountVectorizer()


                matrix = vectorizer.fit_transform(
                    documents
                )


                similarity = cosine_similarity(
                    matrix
                )[0][1]


                match_score = round(
                    similarity * 100,
                    2
                )


            except ValueError:

                match_score = 0


                st.error(
                    "Unable to analyze the provided text. "
                    "Please enter a valid Job Description."
                )


            # Keep score between 0 and 100

            match_score = max(
                0,
                min(match_score, 100)
            )


            # -------------------------------------------------
            # SCREENING RESULT
            # -------------------------------------------------

            st.divider()


            st.subheader(
                "Candidate Screening Result"
            )


            st.metric(
                "Resume Match Score",
                f"{match_score}%"
            )


            st.progress(
                int(match_score)
            )


            # -------------------------------------------------
            # MATCH RESULT MESSAGE
            # -------------------------------------------------

            if match_score >= 70:

                st.success(
                    "Excellent Match! This candidate strongly matches "
                    "the job requirements."
                )


            elif match_score >= 40:

                st.warning(
                    "Average Match. The candidate matches some "
                    "job requirements."
                )


            else:

                st.error(
                    "Low Match. The candidate has limited similarity "
                    "with the job requirements."
                )


            # -------------------------------------------------
            # SKILLS ANALYSIS
            # -------------------------------------------------

            if skills_input.strip():


                skills = [

                    skill.strip().lower()

                    for skill in skills_input.split(",")

                    if skill.strip()

                ]


                resume_lower = resume_text.lower()


                matched_skills = [

                    skill

                    for skill in skills

                    if skill in resume_lower

                ]


                missing_skills = [

                    skill

                    for skill in skills

                    if skill not in resume_lower

                ]


                st.divider()


                st.subheader(
                    "Skills Analysis"
                )


                col1, col2 = st.columns(2)


                # -------------------------------------------------
                # MATCHED SKILLS COLUMN
                # -------------------------------------------------

                with col1:


                    st.success(
                        "Matched Skills"
                    )


                    if matched_skills:

                        for skill in matched_skills:

                            st.write(
                                f"✔ {skill.title()}"
                            )

                    else:

                        st.write(
                            "No matching skills found."
                        )


                # -------------------------------------------------
                # MISSING SKILLS COLUMN
                # -------------------------------------------------

                with col2:


                    st.error(
                        "Missing Skills"
                    )


                    if missing_skills:

                        for skill in missing_skills:

                            st.write(
                                f"✖ {skill.title()}"
                            )

                    else:

                        st.write(
                            "No missing skills found."
                        )


                # -------------------------------------------------
                # SKILL MATCH SCORE
                # -------------------------------------------------

                if skills:

                    skill_score = round(

                        (
                            len(matched_skills)
                            /
                            len(skills)
                        )
                        *
                        100,

                        2

                    )

                else:

                    skill_score = 0


                skill_score = max(
                    0,
                    min(skill_score, 100)
                )


                st.metric(
                    "Required Skills Match",
                    f"{skill_score}%"
                )


                st.progress(
                    int(skill_score)
                )


            # -------------------------------------------------
            # EXTRACTED RESUME TEXT
            # -------------------------------------------------

            st.divider()


            with st.expander(
                "View Extracted Resume Text"
            ):

                st.text(
                    resume_text
                )


            # -------------------------------------------------
            # HOW IT WORKS
            # -------------------------------------------------

            with st.expander(
                "How does Number1CV work?"
            ):


                st.markdown(
                    """

### Resume Processing

1. Upload a PDF resume.
2. Extract text using PyPDF2.
3. Convert the resume and job description into numerical features.
4. Use CountVectorizer to analyze words.
5. Calculate similarity using Cosine Similarity.
6. Check required skills in the resume.
7. Generate Resume Match and Skills Match scores.

                    """
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="footer"><div class="footer-name">Developed by <span>RUTIK TAMBE</span></div><div class="footer-skills">Python • Machine Learning • Data Science • AI</div><div class="footer-tech">Built with Streamlit &amp; Scikit-learn</div></div>',
    unsafe_allow_html=True
)