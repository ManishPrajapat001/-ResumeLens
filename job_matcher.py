import chromadb
from sentence_transformers import SentenceTransformer
from collections import defaultdict

import os
from config import  EMBEDDING_MODEL, TOP_K



# -----------------------------
# 1. Initialize DB + Model
# -----------------------------
def init_db():

    client = chromadb.PersistentClient(path="chromadb_store")

    collection = client.get_or_create_collection(name="resumes")

    return collection


def init_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# 2. Keyword Score
# -----------------------------
def keyword_score(jd, text, metadata_skills):
    jd = jd.lower()
    text = text.lower()

    score = 0

    # skills from metadata
    if metadata_skills:
        skills_list = metadata_skills.split(",")
        for skill in skills_list:
            if skill.strip() in jd:
                score += 1

    # raw text match
    keywords = ["python", "flask", "react", "sql", "docker"]

    for word in keywords:
        if word in jd and word in text:
            score += 1

    return score


# -----------------------------
# 3. Aggregate results
# -----------------------------
def clean_excerpt(text, limit=200):
    return text[:limit] + "..." if len(text) > limit else text
def process_results(results, jd):
    candidates = defaultdict(lambda: {
        "scores": [],
        "texts": [],
        "metadata": None
    })

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        name = meta["name"]

        similarity = 1 - dist  # convert distance → similarity

        candidates[name]["scores"].append(similarity)
        candidates[name]["texts"].append(doc)
        candidates[name]["metadata"] = meta

    final_results = []

    for name, data in candidates.items():
        avg_score = sum(data["scores"]) / max(len(data["scores"]), 1)

        # keyword score boost

        kw_score = keyword_score(jd, " ".join(data["texts"]), data["metadata"]["skills"])

        kw_norm = min(kw_score / 3.0, 1.0)

        # skill overlap bonus (important)

        skills = data["metadata"]["skills"].split(", ")

        jd_lower = jd.lower()

        skill_overlap = sum(1 for s in skills if s.strip() in jd_lower)

        skill_bonus = min(skill_overlap / 3.0, 1.0)

        # combine (balanced)

        final_score = (

            0.5 * avg_score +

            0.3 * kw_norm +

            0.2 * skill_bonus

        )

        # scale nicely (important change)

        final_score = final_score * 0.8 + 0.2   # prevents collapse to near 0

        # clamp

        final_score = max(0.0, min(final_score, 1.0))

        reason = f"Candidate matches skills: {data['metadata']['skills']}"

        final_results.append({
            "candidate_name": name,
            "resume_path": data["metadata"]["path"],
            "match_score": round(final_score * 100, 2),
            "matched_skills": data["metadata"]["skills"].split(", "),
            "relevant_excerpts": [clean_excerpt(t) for t in data["texts"][:2]],
            "reasoning": reason

        })

    final_results.sort(key=lambda x: x["match_score"], reverse=True)

    return final_results


# -----------------------------
# 4. Main Matching Function
# -----------------------------
def match_job(jd_text):
    model = init_model()
    collection = init_db()

    jd_embedding = model.encode(jd_text).tolist()

    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=TOP_K
    )

    final_matches = process_results(results, jd_text)

    return {
        "job_description": jd_text,
        "top_matches": final_matches[:5]
    }


# -----------------------------
# 5. Run
# -----------------------------



def load_job_descriptions(folder_path):

    jds = []

    for file in os.listdir(folder_path):

        if file.endswith(".txt"):

            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:

                content = f.read()

                jds.append((file, content))  # (filename, jd text)

    return jds

if __name__ == "__main__":
    jd_folder = "test_files/jds"

    jds = load_job_descriptions(jd_folder)

    for file_name, jd_text in jds:

        print(f"\n========== Processing {file_name} ==========\n")

        output = match_job(jd_text)

        from pprint import pprint

        pprint(output)




