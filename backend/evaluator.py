import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spacy model
nlp = spacy.load("en_core_web_sm")

def tokenize_and_vectorize(text):
    tokens = [token.text for token in nlp(text.lower())]
    return ' '.join(tokens)

def calculate_cosine_similarity(source_text, target_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([source_text, target_text])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim[0, 1]

def evaluate_jsons(source_json, target_jsons):
    source_mapping = {
        "English_profiency_score": "Required_English_profiency_score",
        "intended_program": "intended_program",
        "intended_degree": "intended_degree",
        "skills": "Required_skills",
        "research_interest": "research_field"
    }

    best_match = None
    max_score = -1

    source_skills_text = tokenize_and_vectorize(source_json["skills"])
    source_research_text = tokenize_and_vectorize(source_json["research_interest"])

    for target_json in target_jsons:
        target_skills_text = tokenize_and_vectorize(target_json["Required_skills"])
        target_research_text = tokenize_and_vectorize(target_json["research_field"])

        score = 0

        # Check if program and degree match
        if source_json["intended_program"].lower() != target_json["intended_program"].lower()  or \
                source_json["intended_degree"].lower()  != target_json["intended_degree"].lower() :
            print("source: " + source_json["intended_program"])
            print("target: " + target_json["intended_program"])

            print("source: " + source_json["intended_degree"])
            print("target: " + target_json["intended_degree"])
            continue

        # Check English proficiency score
        if source_json["English_proficiency_score"] and target_json["Required_English_proficiency_score"]:
            if float(source_json["English_proficiency_score"]) < float(target_json["Required_English_proficiency_score"]):
                continue

        # Calculate cosine similarity for skills and research_interest
        skills_similarity = calculate_cosine_similarity(source_skills_text, target_skills_text)
        research_similarity = calculate_cosine_similarity(source_research_text, target_research_text)

        # Update score based on similarity measures
        score += skills_similarity
        score += research_similarity

        # Update best match if the current score is higher
        if score > max_score:
            max_score = score
            best_match = target_json

    print("Score: : " + str(max_score))
    return best_match if max_score > 0 else None

# Example usage
if __name__ == '__main__':
    # ... (source_json and target_jsons definitions)
    source_json = {
        'English_proficiency_score': '105',
        'intended_program': 'Computer Science',
        'intended_degree': 'PhD',
        'skills': 'Python, C++, C#',
        'research_interest': 'AI in healthcare, quantum computing, machine learning in chemical engineering, cyber security in credit card fraud'
    }
    target_jsons = [
        {
            "university_name": "University of Calgary",
            "professor_name": "Dr. Emily Rogers-Bradley",
            "Required_English_proficiency_score": "",
            "intended_program": "Mechanical Engineering (MME) or Biomedical Engineering (BME)",
            "intended_degree": "MSc or PhD",
            "Required_skills": "",
            "research_field": "robotic prosthesis design, exoskeleton development, and human biomechanics",
        },
        {
            "university_name": "University of Central Florida",
            "professor_name": "Dr. Di Wu",
            "Required_English_proficiency_score": "",
            "intended_program": "Department of ECE",
            "intended_degree": "PhD",
            "Required_skills": "",
            "research_field": "emerging areas of computer architecture, such as unary, approximate, and neuromorphic computing",},
            {
            "university_name": "University of Idaho",
            "professor_name": "Dr. Hasan Jamil",
            "Required_English_proficiency_score": "100",
            "intended_program": "Computer Science",
            "intended_degree": "PhD",
            "Required_skills": "Database, Machine Learning, Deep learning, large language modeling",
            "research_field": "Applications of machine learning and deep learning various fields, large language modeling",},
            {
            "university_name": "University of Alabama",
            "professor_name": "Dr. Golnaz Habibi",
            "Required_English_proficiency_score": "100",
            "intended_program": "Computer Science",
            "intended_degree": "PhD",
            "Required_skills": "C++, Java, Competetive programming, Data Structures and algorithms",
            "research_field": "Software Engineering, Computer Architecture, IOT, Genetic Algorithm",},
            {
            "university_name": "University of Virginia",
            "professor_name": "Dr. Tomonari Furukawa",
            "Required_English_proficiency_score": "",
            "intended_program": "Mechanical and Aerospace Engineering",
            "intended_degree": "Ph.D",
            "Required_skills": "theory of probabilistic robotics, perception, humanoid robotics, robot dynamics/control, mechatronics",
            "research_field": "Humanoid robot design and control, IMU-based humanoid motion tracking, Cooperative mapping and localization, Robot vision based on photometric stereo, Marine robotics",}
                                # ... (other target JSONs)
    ]


    result = evaluate_jsons(source_json, target_jsons)
    print("Best match:", result)
