import os
import re
import secrets
import shutil
import requests
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VECTOR_DB = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# OpenRouter key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<path:name>")
def files(name):
    return send_from_directory(
        UPLOAD_FOLDER,
        name
    )


# ---------------- PDF UPLOAD ----------------

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["pdf_file"]

    filename = secure_filename(
        secrets.token_hex(8)+".pdf"
    )

    path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(path)


    if os.path.exists(VECTOR_DB):
        shutil.rmtree(VECTOR_DB, ignore_errors=True)

    os.makedirs(VECTOR_DB, exist_ok=True)


    docs = PyPDFLoader(path).load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )


    chunks = splitter.split_documents(docs)


    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DB
    )


    return jsonify({
        "filename":filename
    })



# ---------------- OFFLINE CHAT ----------------

@app.route("/chat", methods=["POST"])
def chat():

    question = request.json["message"].lower()

    question = re.sub(
        r"[^a-zA-Z0-9 ]",
        "",
        question
    )


    db = Chroma(
        persist_directory=VECTOR_DB,
        embedding_function=embeddings
    )


    docs = db.as_retriever(
        search_kwargs={"k":4}
    ).invoke(question)



    keywords=[
        x for x in question.split()
        if len(x)>2
    ]


    answer=""

    found=False


    for d in docs:

        if any(
            k in d.page_content.lower()
            for k in keywords
        ):

            found=True

            answer += (
                d.page_content
                +"<br><br>"
            )


    if not found:

        answer = (
        "❌ Information not found in uploaded PDF."
        )


    return jsonify({
        "answer":answer
    })



# ---------------- ONLINE AI ----------------


# ---------------- ONLINE AI ----------------

@app.route("/ai", methods=["POST"])
def ai():

    text = request.json["text"]

    try:

        res = requests.post(

            "https://openrouter.ai/api/v1/chat/completions",

            headers={

                "Authorization":
                f"Bearer {OPENROUTER_API_KEY}",

                "Content-Type":
                "application/json",

                "HTTP-Referer":
                "http://localhost:5000",

                "X-Title":
                "Smart PDF System"
            },


            json={

    "model":
    "deepseek/deepseek-r1-0528",

    "messages":[

        {
            "role":"system",
            "content":"Give simple short paragraph answers"
        },

        {
            "role":"user",
            "content":text
        }

    ],

    "max_tokens":500
},

            timeout=30
        )


        data = res.json()

        print(data)   # IMPORTANT DEBUG


        if "choices" in data:

            answer = data["choices"][0]["message"]["content"]

            return jsonify({
                "extra":answer
            })


        else:

            return jsonify({
                "error":data
            })


    except requests.exceptions.ConnectionError:

        return jsonify({
            "error":"❌ No internet connection"
        })


    except Exception as e:

        print(e)

        return jsonify({
            "error":str(e)
        })



if __name__=="__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT",5000))
    )