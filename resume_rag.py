from http import client
import os
import re
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from config import EMBEDDING_MODEL, COLLECTION_NAME

# load resumes from a folder, support txt and pdf formats, return a list of dicts with text and path
def load_resumes(folder_path):
    resumes = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        text = ""

        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif file.endswith(".pdf"):
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()

        if text.strip():
            resumes.append({
                "text": text,
                "path": file_path
            })

    return resumes


# chunk the resume text into smaller pieces, with a specified chunk size and overlap, return a list of chunks

def chunk_resume(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]

        # alternative way to get chunk words without slicing
        # chunk_words = []
        # for j in range(i, min(i + chunk_size, len(words))):
        #     chunk_words.append(words[j])


        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


# extract metadata from the resume text, such as name, skills, experience, return a dict with the metadata
def extract_metadata(text):

    text_lower = text.lower()

    # Name

    lines = text.split("\n")

    name = "Unknown"

    for line in lines:

        if line.strip():

            name = line.strip()

            break

    # Skills

    skills_list = ["python", "java", "sql", "flask", "react", "docker"]

    skills_found = []

    for skill in skills_list:

        if skill in text_lower:

            skills_found.append(skill)

    # Experience

    experience = 0

    match = re.search(r'(\d+)\+?\s*(years|yrs)', text_lower)

    if match:

        experience = int(match.group(1))

    else:

        word_to_num = {

            "one": 1, "two": 2, "three": 3,

            "four": 4, "five": 5, "six": 6,

            "seven": 7, "eight": 8, "nine": 9,

            "ten": 10

        }

        for word in word_to_num:

            if word + " year" in text_lower:

                experience = word_to_num[word]

                break

    return {

        "name": name,

        "skills": skills_found,

        "experience": experience

    }





# Initialize DB
def init_db():

    client = chromadb.PersistentClient(path="chromadb_store")

    collection = client.get_or_create_collection(name="resumes")

    return collection


#  Initialize model


def init_model():

    return SentenceTransformer(EMBEDDING_MODEL)


# Store in DB


def store_resumes(resumes, collection, model):

    id_counter = 0

    for resume in tqdm(resumes, desc="Processing resumes"):

        text = resume["text"]

        path = resume["path"]

        chunks = chunk_resume(text)

        metadata = extract_metadata(text)

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            collection.add(

                documents=[chunk],

                embeddings=[embedding],

                metadatas=[{

                    "name": metadata["name"],

                    "skills": ", ".join(metadata["skills"]), 

                    "experience": metadata["experience"],

                    "path": path

                }],

                ids=[str(id_counter)]

            )

            id_counter += 1

        
#  Main
if __name__ == "__main__":

    folder_path = "test_files/resumes"

    print("Loading resumes...")

    resumes = load_resumes(folder_path)

    print(f"Loaded {len(resumes)} resumes")

    print("Initializing model and DB...")

    model = init_model()

    collection = init_db()

    print("Storing embeddings...")

    store_resumes(resumes, collection, model)
    

    print("✅ Resume ingestion completed")

    print("Checking stored data...")

    results = collection.get()

    print("Total stored chunks:", len(results["documents"]))
    print("Sample document:", results["documents"][0][:100])
    print("Sample metadata:", results["metadatas"][0])