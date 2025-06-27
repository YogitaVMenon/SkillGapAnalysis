import pandas as pd
import spacy
import ast
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import traceback

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])


class SkillGapAnalysisSystem:
    def __init__(self, job_data_path, resume_data_path, coursera_data_path):
        self.job_data_path = job_data_path
        self.resume_data_path = resume_data_path
        self.coursera_data_path = coursera_data_path

        self.data = None
        self.df = None
        self.coursera_data = None
        self.job_titles_combined = None
        self.skills_and_keywords = None
        self.skills_of_candidate = None
        self.positions_applied = None

        self.load_data()

    def load_data(self):
        self.process_job_data()
        self.process_resume_data()
        self.process_coursera_data()

    def process_job_data(self):
        try:
            # Load job data
            self.data = pd.read_csv(self.job_data_path)
            self.data.columns = self.data.columns.str.strip().str.lower().str.replace(" ", "_")

            self.data['job_titles_combined'] = self.data[['job_title', 'role']].apply(
                lambda x: [item for item in x if pd.notnull(item)], axis=1)

            self.data = self.data.head(10000)

            self.data['extracted_keywords'] = self.extract_action_keywords(
                self.data['responsibilities'].fillna(""))

            self.data['skills_and_keywords'] = self.data.apply(
                lambda row: list(sorted(set(row['skills'].split(', ') + row['extracted_keywords'])))
                if pd.notnull(row['skills']) else sorted(set(row['extracted_keywords'])),
                axis=1
            )

            self.job_titles_combined = self.data['job_titles_combined'].tolist()
            self.skills_and_keywords = self.data['skills_and_keywords'].tolist()

            print(f"Successfully processed {len(self.data)} job descriptions")
        except Exception as e:
            print(f"Error processing job data: {e}")

    def extract_action_keywords(self, texts):
        action_keywords_list = []
        for document in nlp.pipe(texts, batch_size=50):
            action_keywords = [
                " ".join([word.text for word in chunk if word.pos_ not in ["DET", "ADP", "AUX", "CCONJ"]])
                for chunk in document.noun_chunks
                if any(word.pos_ in ["VERB", "NOUN"] for word in chunk)
            ]
            action_keywords_list.append(action_keywords)
        return action_keywords_list

    def process_resume_data(self):
        try:
            self.df = pd.read_csv(self.resume_data_path)

            skills_columns = ["skills", "related_skils_in_job", "responsibilities", "skills_required"]
            positions_columns = ["positions", "extra_curricular_activity_types", "role_positions", "job_position_name"]

            self.df["merged_skills"] = self.df.apply(
                lambda row: list(set(
                    [skill for column in skills_columns for skill in self.process_cell(row[column])]
                )),
                axis=1
            )

            self.df["merged_positions"] = self.df.apply(
                lambda row: list(set(
                    [position for column in positions_columns for position in self.process_cell(row[column])]
                )),
                axis=1
            )

            self.df.rename(columns={"merged_skills": "skills_of_candidate",
                                    "merged_positions": "positions_applied"}, inplace=True)

            self.skills_of_candidate = self.df['skills_of_candidate'].tolist()
            self.positions_applied = self.df['positions_applied'].tolist()

            print(f"Successfully processed {len(self.df)} resumes")
        except Exception as e:
            print(f"Error processing resume data: {e}")

    def process_cell(self, data):
        if pd.isna(data):
            return []
        if isinstance(data, str):
            try:
                evaluated_data = ast.literal_eval(data)
                if isinstance(evaluated_data, list):
                    return [item for sublist in evaluated_data for item in
                            (sublist if isinstance(sublist, list) else [sublist])]
            except (ValueError, SyntaxError):
                return data.split(", ") if ", " in data else data.split()
        return []

    def process_coursera_data(self):
        try:
            self.coursera_data = pd.read_csv(self.coursera_data_path)
            self.coursera_data = self.coursera_data.dropna(subset=['skills'])
            self.coursera_data['embedding'] = self.coursera_data['skills'].apply(
                lambda x: model.encode(str(x), convert_to_tensor=True))

            print(f"Successfully processed {len(self.coursera_data)} Coursera courses")
        except Exception as e:
            print(f"Error processing Coursera data: {e}")

    def analyze_skill_gap(self, candidate_number, interactive=True):
        try:
            candidate_index = candidate_number - 1
            applied_positions = self.positions_applied[candidate_index]
            candidate_skills = set(self.skills_of_candidate[candidate_index])

            if interactive:
                print(f"\nCandidate {candidate_number} applied for these positions:")
                for idx, pos in enumerate(applied_positions):
                    print(f"  {idx + 1}. {pos}")

                selected_index = int(input("\nEnter the number corresponding to the position you want to match: ")) - 1
                if selected_index < 0 or selected_index >= len(applied_positions):
                    print("Invalid selection. Exiting.")
                    return None
            else:
                selected_index = 0

            applied_position = applied_positions[selected_index]

            if interactive:
                print(f"\nSelected Position for Matching: {applied_position}")
                print(f"Candidate's skills: {candidate_skills}\n")

            matched_jobs = {}

            for idx, job_title in enumerate(self.job_titles_combined):
                job_words = set(" ".join(job_title).lower().split()) if isinstance(job_title, list) else set(
                    job_title.lower().split())
                applied_words = set(applied_position.lower().split())

                if job_words & applied_words:
                    required_skills = set(self.skills_and_keywords[idx])
                    missing_skills = required_skills - candidate_skills

                    job_title_str = " / ".join(job_title) if isinstance(job_title, list) else job_title

                    if job_title_str in matched_jobs:
                        matched_jobs[job_title_str]["required_skills"].update(required_skills)
                        matched_jobs[job_title_str]["missing_skills"].update(missing_skills)
                    else:
                        matched_jobs[job_title_str] = {
                            "required_skills": required_skills,
                            "missing_skills": missing_skills
                        }

            if not matched_jobs:
                if interactive:
                    print("\nNo matching jobs found.")
                return None

            unique_jobs_list = list(matched_jobs.keys())

            if interactive:
                print("\nUnique Matched Jobs:")
                for i, job_title in enumerate(unique_jobs_list):
                    print(f"  {i + 1}. {job_title}")

                job_selection = int(input("\nEnter the number of the job you want to analyze: ")) - 1
                if job_selection < 0 or job_selection >= len(unique_jobs_list):
                    print("Invalid selection. Exiting.")
                    return None
            else:
                job_selection = 0

            selected_job_title = unique_jobs_list[job_selection]
            selected_job = matched_jobs[selected_job_title]

            if interactive:
                print(f"\n🔹 Selected Matched Job Title: {selected_job_title}")
                print(f"   ✅ Required Skills: {sorted(selected_job['required_skills'])}")
                print(f"   ❌ Missing Skills: {sorted(selected_job['missing_skills'])}\n")

            recommended_courses = []

            for skill in selected_job['missing_skills']:
                skill_embedding = model.encode(skill, convert_to_tensor=True)
                self.coursera_data['similarity'] = self.coursera_data['embedding'].apply(
                    lambda x: util.cos_sim(skill_embedding, x).item())

                best_matches = self.coursera_data.nlargest(3, 'similarity')
                for _, course in best_matches.iterrows():
                    if interactive:
                        print(
                            f"  - Partner: {course['partner']} | Course: {course['course']} | Similarity Score: {course['similarity']:.4f}")

                    recommended_courses.append({
                        'partner': course['partner'],
                        'course': course['course'],
                        'similarity': course['similarity']
                    })

            results = {
                'candidate_number': candidate_number,
                'applied_position': applied_position,
                'candidate_skills': list(candidate_skills),
                'selected_job_title': selected_job_title,
                'required_skills': list(selected_job['required_skills']),
                'missing_skills': list(selected_job['missing_skills']),
                'recommended_courses': recommended_courses
            }

            return results

        except Exception as e:
            print(f"Error in skill gap analysis: {e}")
            return None

    def get_missing_skills(self, candidate_number, position_index=0):
        if hasattr(self, 'test_data') and not self.test_data.empty:
            if candidate_number - 1 >= len(self.test_data):
                return []

            applied_position = self.test_data.iloc[candidate_number - 1]['position_applied']

        else:
            candidate_index = candidate_number - 1
            if candidate_index < 0 or candidate_index >= len(self.positions_applied):
                return []

            applied_positions = self.positions_applied[candidate_index]
            if position_index < 0 or position_index >= len(applied_positions):
                return []

            applied_position = applied_positions[position_index]
            candidate_skills = set(self.skills_of_candidate[candidate_index])

        if pd.isna(applied_position):
            return []

        applied_words = set(applied_position.lower().split())

        all_required_skills = set()
        for idx, job_title in enumerate(self.job_titles_combined):
            job_words = set(" ".join(job_title).lower().split()) if isinstance(job_title, list) else set(
                job_title.lower().split())
            if job_words & applied_words:
                all_required_skills.update(self.skills_and_keywords[idx])

        missing_skills = all_required_skills - candidate_skills
        return list(missing_skills)

    def get_recommended_courses(self, missing_skills):
        recommended_courses = []

        for skill in missing_skills:
            skill_embedding = model.encode(skill, convert_to_tensor=True)
            self.coursera_data['similarity'] = self.coursera_data['embedding'].apply(
                lambda x: util.cos_sim(skill_embedding, x).item())

            best_matches = self.coursera_data.nlargest(1, 'similarity')
            for _, course in best_matches.iterrows():
                recommended_courses.append(f"{course['partner']}: {course['course']}")

        return recommended_courses


class SkillGapSystemEvaluator:
    def __init__(self, system, test_data=None, ground_truth=None):
        self.system = system
        self.test_data = test_data
        self.ground_truth = ground_truth

    def create_evaluation_dataset(self, num_candidates=50):
        all_skills = [
            'python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'nosql',
            'react', 'angular', 'node.js', 'django', 'flask', 'docker', 'kubernetes',
            'aws', 'gcp', 'azure', 'data analysis', 'machine learning', 'deep learning',
            'natural language processing', 'computer vision', 'statistics', 'calculus',
            'linear algebra', 'database design', 'data modeling', 'data visualization',
            'tableau', 'power bi', 'excel', 'project management', 'agile', 'scrum'
        ]

        all_courses = [
            'Python Programming', 'Java Development', 'C++ Fundamentals',
            'JavaScript for Web Development', 'HTML and CSS', 'SQL Database Management',
            'NoSQL Database Systems', 'React.js', 'Angular Framework', 'Node.js Development',
            'Django Web Framework', 'Flask API Development', 'Docker Containers',
            'Kubernetes Orchestration', 'AWS Cloud Services', 'Google Cloud Platform',
            'Microsoft Azure', 'Data Analysis with Python', 'Machine Learning Fundamentals',
            'Deep Learning with TensorFlow', 'NLP with Python', 'Computer Vision Applications',
            'Statistics for Data Science', 'Calculus for Machine Learning',
            'Linear Algebra Essentials', 'Database Design Principles', 'Data Modeling Techniques',
            'Data Visualization Best Practices', 'Tableau Dashboarding', 'Power BI Analytics',
            'Advanced Excel', 'Project Management', 'Agile Development', 'Scrum Master Certification'
        ]

        all_positions = [
            'Software Engineer', 'Data Scientist', 'Web Developer', 'Frontend Developer',
            'Backend Developer', 'Full Stack Developer', 'DevOps Engineer', 'Cloud Engineer',
            'Machine Learning Engineer', 'AI Researcher', 'Database Administrator',
            'Data Analyst', 'Business Intelligence Analyst', 'Project Manager'
        ]

        candidate_ids = list(range(1, num_candidates + 1))
        test_data = {
            'candidate_id': candidate_ids,
            'position_applied': [random.choice(all_positions) for _ in range(num_candidates)]
        }
        ground_truth = {
            'candidate_id': [],
            'position_applied': [],
            'required_skills': [],
            'missing_skills': [],
            'recommended_courses': []
        }

        for cid, position in zip(test_data['candidate_id'], test_data['position_applied']):
            if 'Software' in position or 'Developer' in position:
                req_skills = random.sample(all_skills[:15], random.randint(5, 10))
            elif 'Data' in position or 'Machine' in position or 'AI' in position:
                req_skills = random.sample(all_skills[15:], random.randint(5, 10))
            else:
                req_skills = random.sample(all_skills, random.randint(5, 10))

            missing_skills = random.sample(req_skills, random.randint(2, min(5, len(req_skills))))

            rec_courses = []
            for skill in missing_skills:
                matching_courses = [course for course in all_courses if skill.lower() in course.lower()]
                if matching_courses:
                    rec_courses.append(random.choice(matching_courses))

            if not rec_courses or len(rec_courses) < len(missing_skills):
                additional_courses = random.sample(all_courses, max(1, len(missing_skills) - len(rec_courses)))
                rec_courses.extend(additional_courses)
                rec_courses = list(set(rec_courses))[:len(missing_skills)]

            ground_truth['candidate_id'].append(cid)
            ground_truth['position_applied'].append(position)
            ground_truth['required_skills'].append(req_skills)
            ground_truth['missing_skills'].append(missing_skills)
            ground_truth['recommended_courses'].append(rec_courses)

        test_df = pd.DataFrame(test_data)
        ground_truth_df = pd.DataFrame(ground_truth)

        self.test_data = test_df
        self.ground_truth = ground_truth_df

        return test_df, ground_truth_df

    def evaluate(self):
        if self.test_data is None or self.ground_truth is None:
            print("Test data or ground truth not available. Creating synthetic data...")
            self.create_evaluation_dataset()

        results = {
            'candidate_id': [],
            'position_applied': [],
            'predicted_skills_needed': [],
            'actual_skills_needed': [],
            'predicted_courses': [],
            'actual_courses': [],
            'skill_precision': [],
            'skill_recall': [],
            'skill_f1': [],
            'course_relevance_score': [],
            'satisfaction_score': []
        }

        for _, candidate in self.test_data.iterrows():
            candidate_id = candidate['candidate_id']
            position = candidate['position_applied']

            gt = self.ground_truth[(self.ground_truth['candidate_id'] == candidate_id) &
                                   (self.ground_truth['position_applied'] == position)]

            if gt.empty:
                continue

            gt = gt.iloc[0]

            predicted_skills = self.system.get_missing_skills(candidate_id)
            actual_skills = gt['missing_skills']
            precision, recall, f1 = self.calculate_skill_metrics(predicted_skills, actual_skills)
            predicted_courses = self.system.get_recommended_courses(predicted_skills)
            actual_courses = gt['recommended_courses']
            course_relevance = self.calculate_course_relevance(predicted_courses, actual_courses)
            satisfaction = self.calculate_satisfaction_score(precision, recall, course_relevance)
            results['candidate_id'].append(candidate_id)
            results['position_applied'].append(position)
            results['predicted_skills_needed'].append(predicted_skills)
            results['actual_skills_needed'].append(actual_skills)
            results['predicted_courses'].append(predicted_courses)
            results['actual_courses'].append(actual_courses)
            results['skill_precision'].append(precision)
            results['skill_recall'].append(recall)
            results['skill_f1'].append(f1)
            results['course_relevance_score'].append(course_relevance)
            results['satisfaction_score'].append(satisfaction)
        results_df = pd.DataFrame(results)
        overall_metrics = {
            'avg_skill_precision': results_df['skill_precision'].mean(),
            'avg_skill_recall': results_df['skill_recall'].mean(),
            'avg_skill_f1': results_df['skill_f1'].mean(),
            'avg_course_relevance': results_df['course_relevance_score'].mean(),
            'avg_satisfaction': results_df['satisfaction_score'].mean()
        }

        cm = self.generate_confusion_matrix(results_df)

        return results_df, overall_metrics, cm

    def calculate_skill_metrics(self, predicted_skills, actual_skills):
        pred_set = set(predicted_skills)
        actual_set = set(actual_skills)

        tp = len(pred_set.intersection(actual_set))
        fp = len(pred_set - actual_set)
        fn = len(actual_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def calculate_course_relevance(self, predicted_courses, actual_courses):
        if not predicted_courses or not actual_courses:
            return 0
        similarities = []
        for pred_course in predicted_courses:
            course_sims = []
            pred_embedding = model.encode(pred_course, convert_to_tensor=True)

            for actual_course in actual_courses:
                actual_embedding = model.encode(actual_course, convert_to_tensor=True)
                sim = util.cos_sim(pred_embedding, actual_embedding).item()
                course_sims.append(sim)

            similarities.append(max(course_sims) if course_sims else 0)

        return sum(similarities) / len(similarities) if similarities else 0

    def calculate_satisfaction_score(self, precision, recall, course_relevance, weights=(0.3, 0.3, 0.4)):
        return weights[0] * precision + weights[1] * recall + weights[2] * course_relevance

    def generate_confusion_matrix(self, results_df):
        all_predicted = set()
        all_actual = set()
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0

        for pred, actual in zip(results_df['predicted_skills_needed'], results_df['actual_skills_needed']):
            all_predicted.update(pred)
            all_actual.update(actual)

        all_skills = all_predicted.union(all_actual)

        for pred, actual in zip(results_df['predicted_skills_needed'], results_df['actual_skills_needed']):
            pred_set = set(pred)
            actual_set = set(actual)

            for skill in all_skills:
                if skill in pred_set and skill in actual_set:
                    true_pos += 1
                elif skill in pred_set and skill not in actual_set:
                    false_pos += 1
                elif skill not in pred_set and skill in actual_set:
                    false_neg += 1
                else:
                    true_neg += 1

        cm = np.array([
            [true_pos, false_neg],
            [false_pos, true_neg]
        ])

        return cm

    def visualize_evaluation(self, results_df, overall_metrics, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Needed', 'Predicted Not Needed'],
                    yticklabels=['Actually Needed', 'Actually Not Needed'])
        plt.title('Confusion Matrix for Skills Prediction')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Saved confusion matrix visualization to 'confusion_matrix.png'")

        plt.figure(figsize=(12, 6))
        x = range(len(results_df))
        plt.plot(x, results_df['skill_precision'], 'o-', label='Precision')
        plt.plot(x, results_df['skill_recall'], 's-', label='Recall')
        plt.plot(x, results_df['course_relevance_score'], 'p-', label='Course Relevance')
        plt.plot(x, results_df['satisfaction_score'], 'D-', label='Satisfaction')
        plt.xlabel('Candidate Index')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics by Candidate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('metrics_by_candidate.png')
        print("Saved metrics visualization to 'metrics_by_candidate.png'")

        plt.figure(figsize=(10, 6))
        metrics = list(overall_metrics.keys())
        values = list(overall_metrics.values())

        plt.bar(metrics, values, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Overall System Performance Metrics')
        plt.ylim(0, 1)

        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

        plt.tight_layout()
        plt.savefig('overall_metrics.png')
        print("Saved overall metrics visualization to 'overall_metrics.png'")


def main():
    job_data_path = r"D:\Research\dataset\job_descriptions.csv"
    resume_data_path = r"D:\Research\dataset\resume_data.csv"
    coursera_data_path = r"D:\Coursera.csv"

    print("\n===== Skill Gap Analysis System Evaluation =====\n")

    print("Choose an option:")
    print("1. Run the system interactively for a single candidate")
    print("2. Evaluate system performance with synthetic data")

    choice = input("\nEnter your choice (1/2): ")

    try:
        print("\nInitializing Skill Gap Analysis System...")
        system = SkillGapAnalysisSystem(job_data_path, resume_data_path, coursera_data_path)

        if choice == '1':
            candidate_number = int(input("\nEnter candidate number: "))
            system.analyze_skill_gap(candidate_number, interactive=True)

        elif choice == '2':
            print("\nEvaluating system performance with synthetic data...")
            evaluator = SkillGapSystemEvaluator(system)

            num_candidates = int(input("\nEnter number of synthetic candidates to generate (recommended 10-50): "))
            test_data, ground_truth = evaluator.create_evaluation_dataset(num_candidates=num_candidates)

            test_data.to_csv('synthetic_test_data.csv', index=False)
            ground_truth.to_csv('synthetic_ground_truth.csv', index=False)
            print("\nGenerated synthetic data and saved to CSV files.")

            try:
                print("\nRunning evaluation...")
                results_df, overall_metrics, cm = evaluator.evaluate()
            except Exception as e:
                print("Detailed error info:")
                traceback.print_exc()

            results_df.to_csv('evaluation_results.csv', index=False)

            print("\n===== Overall System Performance =====")
            for metric, value in overall_metrics.items():
                print(f"{metric}: {value:.4f}")

            print("\n===== Confusion Matrix =====")
            print(cm)

            print("\nCreating visualizations...")
            evaluator.visualize_evaluation(results_df, overall_metrics, cm)


        else:
            print("Invalid choice. Please run the script again and choose 1 or 2.")

    except Exception as e:
         print(f"An error occurred: {e}")

    print("\nEvaluation complete! Results saved to CSV and visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
