"""Advanced skill matching and similarity detection."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import logging
from collections import defaultdict
from difflib import SequenceMatcher

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class SkillMatcher:
    """Advanced skill matching for deduplication and standardization."""

    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.8
        self.skill_clusters = {}
        self.canonical_skills = {}

    def find_similar_skills(
        self,
        skills_list: List[str],
        similarity_threshold: float = 0.8
    ) -> Dict[str, List[str]]:
        """Find groups of similar skills for standardization."""
        logger.info(f"Finding similar skills among {len(skills_list)} skills...")

        # Get embeddings for all skills
        embeddings = self.sentence_model.encode(skills_list)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Find similar skill groups
        similar_groups = defaultdict(list)
        processed_skills = set()

        for i, skill1 in enumerate(skills_list):
            if skill1 in processed_skills:
                continue

            similar_skills = [skill1]

            for j, skill2 in enumerate(skills_list[i+1:], i+1):
                if similarity_matrix[i][j] >= similarity_threshold:
                    similar_skills.append(skill2)
                    processed_skills.add(skill2)

            if len(similar_skills) > 1:
                # Use the most common or longest skill as the canonical form
                canonical_skill = max(similar_skills, key=len)
                similar_groups[canonical_skill] = similar_skills

            processed_skills.add(skill1)

        logger.info(f"Found {len(similar_groups)} groups of similar skills")
        return dict(similar_groups)

    def standardize_skill_names(
        self,
        skills_df: pd.DataFrame,
        skill_column: str = "skill_name"
    ) -> pd.DataFrame:
        """Standardize skill names by grouping similar variations."""
        logger.info("Standardizing skill names...")

        # Get unique skills
        unique_skills = skills_df[skill_column].unique().tolist()

        # Find similar groups
        similar_groups = self.find_similar_skills(unique_skills)

        # Create mapping from variations to canonical forms
        skill_mapping = {}
        for canonical, variations in similar_groups.items():
            for variation in variations:
                skill_mapping[variation] = canonical

        # Apply standardization
        standardized_df = skills_df.copy()
        standardized_df[f"{skill_column}_standardized"] = standardized_df[skill_column].map(
            lambda x: skill_mapping.get(x, x)
        )

        # Log statistics
        original_count = len(unique_skills)
        standardized_count = len(standardized_df[f"{skill_column}_standardized"].unique())
        logger.info(f"Standardized {original_count} skills to {standardized_count} canonical forms")

        return standardized_df

    def detect_skill_clusters(
        self,
        skills_df: pd.DataFrame,
        min_samples: int = 3,
        eps: float = 0.3
    ) -> pd.DataFrame:
        """Detect clusters of related skills using DBSCAN."""
        logger.info("Detecting skill clusters...")

        # Get unique skills and their embeddings
        unique_skills = skills_df["skill_name"].unique()
        embeddings = self.sentence_model.encode(unique_skills)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)

        # Create skill-to-cluster mapping
        skill_to_cluster = dict(zip(unique_skills, cluster_labels))

        # Add cluster information to DataFrame
        clustered_df = skills_df.copy()
        clustered_df["skill_cluster"] = clustered_df["skill_name"].map(skill_to_cluster)

        # Name clusters based on most frequent skill
        cluster_names = {}
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise points
                cluster_names[cluster_id] = "Unclustered"
            else:
                cluster_skills = clustered_df[clustered_df["skill_cluster"] == cluster_id]
                most_common_skill = cluster_skills["skill_name"].value_counts().index[0]
                cluster_names[cluster_id] = f"Cluster_{cluster_id}_{most_common_skill}"

        clustered_df["cluster_name"] = clustered_df["skill_cluster"].map(cluster_names)

        logger.info(f"Detected {len(cluster_names) - 1} skill clusters")  # -1 for noise

        return clustered_df

    def calculate_skill_similarity_matrix(
        self,
        skills_list: List[str]
    ) -> pd.DataFrame:
        """Calculate pairwise similarity matrix for skills."""
        logger.info(f"Calculating similarity matrix for {len(skills_list)} skills...")

        # Get embeddings
        embeddings = self.sentence_model.encode(skills_list)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Create DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=skills_list,
            columns=skills_list
        )

        return similarity_df

    def find_complementary_skills(
        self,
        target_skill: str,
        skills_df: pd.DataFrame,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find skills that frequently appear together with a target skill."""
        logger.info(f"Finding complementary skills for: {target_skill}")

        # Get jobs that mention the target skill
        target_jobs = skills_df[skills_df["skill_name"] == target_skill]["job_id"].unique()

        if len(target_jobs) == 0:
            logger.warning(f"No jobs found with skill: {target_skill}")
            return []

        # Get all skills mentioned in those jobs
        complementary_skills = skills_df[
            (skills_df["job_id"].isin(target_jobs)) &
            (skills_df["skill_name"] != target_skill)
        ]

        # Calculate co-occurrence frequency
        skill_counts = complementary_skills["skill_name"].value_counts()
        total_target_jobs = len(target_jobs)

        # Calculate complementary score (frequency / total jobs with target skill)
        complementary_scores = [
            (skill, count / total_target_jobs)
            for skill, count in skill_counts.head(top_k).items()
        ]

        logger.info(f"Found {len(complementary_scores)} complementary skills for {target_skill}")

        return complementary_scores

    def create_skill_network(
        self,
        skills_df: pd.DataFrame,
        min_co_occurrence: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Create a network of skill relationships based on co-occurrence."""
        logger.info("Creating skill co-occurrence network...")

        # Group skills by job
        job_skills = skills_df.groupby("job_id")["skill_name"].apply(list).to_dict()

        # Calculate co-occurrence matrix
        skill_network = defaultdict(lambda: defaultdict(int))

        for job_id, skills in job_skills.items():
            # Count co-occurrences for each pair of skills in the job
            for i, skill1 in enumerate(skills):
                for j, skill2 in enumerate(skills[i+1:], i+1):
                    skill_network[skill1][skill2] += 1
                    skill_network[skill2][skill1] += 1

        # Filter by minimum co-occurrence and normalize
        filtered_network = {}
        for skill1, connections in skill_network.items():
            filtered_connections = {}
            for skill2, count in connections.items():
                if count >= min_co_occurrence:
                    # Normalize by total occurrences of skill1
                    skill1_total = len(skills_df[skills_df["skill_name"] == skill1])
                    normalized_strength = count / skill1_total
                    filtered_connections[skill2] = normalized_strength

            if filtered_connections:
                filtered_network[skill1] = filtered_connections

        logger.info(f"Created network with {len(filtered_network)} nodes")

        return filtered_network