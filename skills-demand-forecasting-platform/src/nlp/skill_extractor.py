"""Advanced skill extraction from job descriptions using NLP techniques."""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Set, Tuple, Optional, Any
import logging
from collections import Counter, defaultdict
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Using basic skill extraction.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using rule-based extraction only.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some features may be limited.")

from src.utils.config_manager import config
from src.utils.file_utils import FileHandler

logger = logging.getLogger(__name__)

class SkillExtractor:
    """Advanced skill extraction using hybrid NLP approaches."""

    def __init__(self):
        self.nlp_config = config.get_model_config("nlp")
        self.confidence_threshold = self.nlp_config.get("skill_extraction", {}).get("confidence_threshold", 0.7)

        # Load models with fallbacks
        self._load_models()

        # Load skill taxonomies
        self.tech_skills = self._load_tech_skills()
        self.soft_skills = self._load_soft_skills()
        self.skill_patterns = self._create_skill_patterns()

    def _load_models(self):
        """Load NLP models for skill extraction with fallbacks."""
        self.nlp = None
        self.sentence_model = None

        if SPACY_AVAILABLE:
            try:
                model_name = self.nlp_config.get("models", {}).get("spacy_model", "en_core_web_sm")
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")
                self.nlp = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                transformer_name = self.nlp_config.get("models", {}).get("sentence_transformer", "all-MiniLM-L6-v2")
                self.sentence_model = SentenceTransformer(transformer_name)
                logger.info(f"Loaded sentence transformer: {transformer_name}")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_model = None

    def _load_tech_skills(self) -> Set[str]:
        """Load comprehensive technology skills taxonomy."""
        return {
            # Programming Languages
            "Python", "JavaScript", "Java", "C++", "C#", "TypeScript", "Go", "Rust", "Swift", "Kotlin",
            "Ruby", "PHP", "Scala", "R", "MATLAB", "Julia", "Perl", "Haskell", "Clojure", "Dart",

            # Web Technologies
            "HTML", "CSS", "React", "Vue.js", "Angular", "Node.js", "Express.js", "Django", "Flask",
            "Spring Boot", "ASP.NET", "Laravel", "Ruby on Rails", "Next.js", "Nuxt.js", "Svelte",

            # Databases
            "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
            "Oracle", "SQLite", "MariaDB", "DynamoDB", "Neo4j", "InfluxDB", "CouchDB",

            # Cloud Platforms
            "AWS", "Google Cloud", "Microsoft Azure", "IBM Cloud", "DigitalOcean", "Heroku",
            "Vercel", "Netlify", "Firebase", "Supabase",

            # DevOps & Tools
            "Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions", "Terraform",
            "Ansible", "Chef", "Puppet", "Vagrant", "Prometheus", "Grafana", "ELK Stack",

            # Data Science & ML
            "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "Matplotlib", "Seaborn",
            "Jupyter", "Apache Spark", "Hadoop", "Kafka", "Airflow", "MLflow", "Kubeflow",

            # Mobile Development
            "React Native", "Flutter", "Xamarin", "Ionic", "Cordova", "SwiftUI", "Android SDK",

            # Version Control
            "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Mercurial",

            # Testing
            "Jest", "Pytest", "JUnit", "Selenium", "Cypress", "Mocha", "Chai", "TestNG",

            # Methodologies
            "Agile", "Scrum", "Kanban", "DevOps", "CI/CD", "TDD", "BDD", "Microservices",
        }

    def _load_soft_skills(self) -> Set[str]:
        """Load soft skills taxonomy."""
        return {
            "Leadership", "Communication", "Teamwork", "Problem Solving", "Critical Thinking",
            "Project Management", "Time Management", "Adaptability", "Creativity", "Innovation",
            "Analytical Thinking", "Decision Making", "Negotiation", "Mentoring", "Training",
            "Customer Service", "Sales", "Marketing", "Strategic Planning", "Business Analysis"
        }

    def _create_skill_patterns(self) -> List[Dict[str, Any]]:
        """Create regex patterns for skill detection."""
        patterns = []

        # Technology patterns
        for skill in self.tech_skills:
            # Exact match pattern
            patterns.append({
                "skill": skill,
                "pattern": re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE),
                "type": "technical",
                "confidence": 0.9
            })

        # Common variations and abbreviations
        variations = {
            "JavaScript": ["JS", "Javascript", "ECMAScript"],
            "TypeScript": ["TS"],
            "Python": ["Python3", "Py"],
            "C++": ["CPP", "C Plus Plus"],
            "C#": ["CSharp", "C Sharp"],
            "PostgreSQL": ["Postgres", "PostGIS"],
            "MongoDB": ["Mongo"],
            "React": ["ReactJS", "React.js"],
            "Vue.js": ["Vue", "VueJS"],
            "Angular": ["AngularJS", "Angular2+"],
            "Node.js": ["NodeJS", "Node"],
            "Amazon Web Services": ["AWS"],
            "Google Cloud Platform": ["GCP", "Google Cloud"],
            "Microsoft Azure": ["Azure"],
        }

        for main_skill, vars in variations.items():
            for var in vars:
                patterns.append({
                    "skill": main_skill,
                    "pattern": re.compile(r'\b' + re.escape(var) + r'\b', re.IGNORECASE),
                    "type": "technical",
                    "confidence": 0.8
                })

        return patterns

    def extract_skills_from_text(self, text: str, method: str = "hybrid") -> Dict[str, Any]:
        """Extract skills from text using specified method."""
        if pd.isna(text) or not text.strip():
            return {"skills": [], "confidence_scores": {}}

        if method == "hybrid":
            return self._extract_skills_hybrid(text)
        elif method == "rule_based":
            return self._extract_skills_rule_based(text)
        elif method == "ml_based":
            return self._extract_skills_ml_based(text)
        else:
            # Fallback to rule-based
            return self._extract_skills_rule_based(text)

    def _extract_skills_hybrid(self, text: str) -> Dict[str, Any]:
        """Hybrid approach combining rule-based and ML methods."""
        # Get results from both methods
        rule_results = self._extract_skills_rule_based(text)

        if self.sentence_model:
            ml_results = self._extract_skills_ml_based(text)
        else:
            ml_results = {"skills": [], "confidence_scores": {}}

        # Combine and weight the results
        combined_skills = {}

        # Add rule-based results with higher confidence
        for skill in rule_results["skills"]:
            combined_skills[skill] = rule_results["confidence_scores"].get(skill, 0.8)

        # Add ML-based results, boosting confidence if already found by rules
        for skill in ml_results["skills"]:
            if skill in combined_skills:
                # Boost confidence for skills found by both methods
                combined_skills[skill] = min(1.0, combined_skills[skill] + 0.15)
            else:
                combined_skills[skill] = ml_results["confidence_scores"].get(skill, 0.6)

        # Filter by confidence threshold
        filtered_skills = {
            skill: conf for skill, conf in combined_skills.items()
            if conf >= self.confidence_threshold
        }

        return {
            "skills": list(filtered_skills.keys()),
            "confidence_scores": filtered_skills,
            "method": "hybrid"
        }

    def _extract_skills_rule_based(self, text: str) -> Dict[str, Any]:
        """Extract skills using rule-based pattern matching."""
        found_skills = {}

        # Apply regex patterns
        for pattern_info in self.skill_patterns:
            matches = pattern_info["pattern"].findall(text)
            if matches:
                skill = pattern_info["skill"]
                confidence = pattern_info["confidence"]

                # Boost confidence for multiple mentions
                mention_count = len(matches)
                boosted_confidence = min(1.0, confidence + (mention_count - 1) * 0.05)

                found_skills[skill] = boosted_confidence

        # Also check for exact matches in skill lists
        text_lower = text.lower()
        for skill in self.tech_skills.union(self.soft_skills):
            if skill.lower() in text_lower:
                if skill not in found_skills:
                    found_skills[skill] = 0.85

        return {
            "skills": list(found_skills.keys()),
            "confidence_scores": found_skills,
            "method": "rule_based"
        }

    def _extract_skills_ml_based(self, text: str) -> Dict[str, Any]:
        """Extract skills using ML-based semantic similarity."""
        found_skills = {}

        if not self.sentence_model or not SKLEARN_AVAILABLE:
            return {
                "skills": [],
                "confidence_scores": {},
                "method": "ml_based"
            }

        try:
            # Get text embedding
            text_embedding = self.sentence_model.encode([text])

            # Get embeddings for all known skills
            all_skills = list(self.tech_skills.union(self.soft_skills))
            skill_embeddings = self.sentence_model.encode(all_skills)

            # Calculate similarities
            similarities = cosine_similarity(text_embedding, skill_embeddings)[0]

            # Find skills above similarity threshold
            similarity_threshold = 0.3  # Lower threshold for semantic matching
            for i, similarity in enumerate(similarities):
                if similarity > similarity_threshold:
                    skill = all_skills[i]
                    # Convert similarity to confidence score
                    confidence = min(0.95, similarity * 1.5)  # Scale and cap
                    found_skills[skill] = confidence

        except Exception as e:
            logger.warning(f"ML-based extraction failed: {e}")

        return {
            "skills": list(found_skills.keys()),
            "confidence_scores": found_skills,
            "method": "ml_based"
        }

    def extract_skills_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "description",
        method: str = "hybrid"
    ) -> pd.DataFrame:
        """Extract skills from a DataFrame of job postings."""
        logger.info(f"Extracting skills from {len(df)} job postings using {method} method...")

        results = []

        for idx, row in df.iterrows():
            job_id = row.get("job_id", f"job_{idx}")
            text = row.get(text_column, "")

            # Extract skills
            extraction_result = self.extract_skills_from_text(text, method)

            # Create records for each skill
            for skill in extraction_result["skills"]:
                results.append({
                    "job_id": job_id,
                    "skill_name": skill,
                    "confidence": extraction_result["confidence_scores"].get(skill, 0.0),
                    "extraction_method": extraction_result.get("method", method),
                    "skill_type": self._classify_skill_type(skill)
                })

            # Log progress
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} job postings")

        skills_df = pd.DataFrame(results)
        logger.info(f"Extracted {len(skills_df)} skill mentions from {len(df)} jobs")

        return skills_df

    def _classify_skill_type(self, skill: str) -> str:
        """Classify skill as technical, soft, or industry-specific."""
        if skill in self.tech_skills:
            return "technical"
        elif skill in self.soft_skills:
            return "soft"
        else:
            return "industry"

    def create_skill_taxonomy(self, skills_df: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive skill taxonomy with categories."""
        logger.info("Creating skill taxonomy...")

        if skills_df.empty:
            return pd.DataFrame()

        # Group skills and calculate statistics
        skill_stats = skills_df.groupby("skill_name").agg({
            "job_id": "count",
            "confidence": ["mean", "std"],
            "skill_type": "first"
        }).round(3)

        # Flatten column names
        skill_stats.columns = ["frequency", "avg_confidence", "confidence_std", "skill_type"]
        skill_stats = skill_stats.reset_index()

        # Handle NaN values
        skill_stats["confidence_std"] = skill_stats["confidence_std"].fillna(0)

        # Add skill categories
        skill_stats["category"] = skill_stats["skill_name"].apply(self._categorize_skill)

        # Calculate relative frequency (percentage)
        total_jobs = skills_df["job_id"].nunique() if not skills_df.empty else 1
        skill_stats["frequency_percentage"] = (skill_stats["frequency"] / total_jobs * 100).round(2)

        # Sort by frequency
        skill_stats = skill_stats.sort_values("frequency", ascending=False)

        logger.info(f"Created taxonomy for {len(skill_stats)} unique skills")

        return skill_stats

    def _categorize_skill(self, skill: str) -> str:
        """Categorize skills into detailed categories."""
        skill_lower = skill.lower()

        # Programming languages
        programming_langs = ["python", "javascript", "java", "c++", "c#", "typescript", "go", "rust", "swift", "kotlin", "ruby", "php", "scala", "r"]
        if any(lang in skill_lower for lang in programming_langs):
            return "Programming Languages"

        # Web frameworks
        web_frameworks = ["react", "vue", "angular", "node", "django", "flask", "spring", "laravel", "rails"]
        if any(fw in skill_lower for fw in web_frameworks):
            return "Web Frameworks"

        # Databases
        databases = ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", "cassandra"]
        if any(db in skill_lower for db in databases):
            return "Databases"

        # Cloud platforms
        cloud_platforms = ["aws", "azure", "google cloud", "gcp", "heroku", "digitalocean"]
        if any(cloud in skill_lower for cloud in cloud_platforms):
            return "Cloud Platforms"

        # DevOps tools
        devops_tools = ["docker", "kubernetes", "jenkins", "terraform", "ansible", "chef", "puppet"]
        if any(tool in skill_lower for tool in devops_tools):
            return "DevOps & Infrastructure"

        # Data science
        data_science = ["tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "spark", "hadoop", "kafka"]
        if any(ds in skill_lower for ds in data_science):
            return "Data Science & ML"

        # Mobile development
        mobile = ["react native", "flutter", "xamarin", "ionic", "swift", "android"]
        if any(mob in skill_lower for mob in mobile):
            return "Mobile Development"

        # Soft skills
        soft_skills = ["leadership", "communication", "teamwork", "problem solving", "project management"]
        if any(soft in skill_lower for soft in soft_skills):
            return "Soft Skills"

        return "Other"
