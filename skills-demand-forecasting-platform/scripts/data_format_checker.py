"""
Quick checker to verify your data formats match expected schemas.
Run this first before full validation to catch format issues early.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_linkedin_124k_format(data_path: str) -> Dict[str, Any]:
    """Check LinkedIn 124k data format."""

    results = {"status": "unknown", "issues": [], "files_checked": {}}

    expected_files = {
        "postings.csv": {
            "required_columns": ["job_id", "title", "company_id"],
            "optional_columns": ["description", "location", "posted_date"],
            "min_rows": 100
        },
        "companies.csv": {
            "required_columns": ["company_id", "name"],
            "optional_columns": ["description"],
            "min_rows": 10
        },
        "salaries.csv": {
            "required_columns": ["job_id"],
            "optional_columns": ["min_salary", "max_salary", "med_salary"],
            "min_rows": 50
        },
        "job_skills.csv": {
            "required_columns": ["job_id", "skill_abr"],
            "optional_columns": [],
            "min_rows": 200
        }
    }

    base_path = Path(data_path)

    for filename, requirements in expected_files.items():
        file_path = base_path / filename

        if not file_path.exists():
            results["issues"].append(f"❌ Missing file: {filename}")
            continue

        try:
            # Check file content
            df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows

            file_results = {
                "exists": True,
                "rows": len(df),
                "columns": list(df.columns),
                "issues": []
            }

            # Check required columns
            missing_required = set(requirements["required_columns"]) - set(df.columns)
            if missing_required:
                file_results["issues"].append(f"Missing required columns: {list(missing_required)}")

            # Check minimum rows
            if len(df) < requirements["min_rows"]:
                file_results["issues"].append(f"Too few rows: {len(df)} < {requirements['min_rows']}")

            # Specific checks
            if filename == "postings.csv":
                if "description" in df.columns:
                    empty_descriptions = df["description"].isnull().sum()
                    if empty_descriptions > len(df) * 0.5:
                        file_results["issues"].append("More than 50% of descriptions are empty")

            if filename == "salaries.csv":
                salary_cols = [col for col in df.columns if "salary" in col.lower()]
                if salary_cols:
                    for col in salary_cols:
                        if df[col].dtype not in ['int64', 'float64']:
                            file_results["issues"].append(f"Salary column {col} is not numeric")

            results["files_checked"][filename] = file_results

            if not file_results["issues"]:
                logger.info(f"✅ {filename}: Format looks good ({len(df)} rows)")
            else:
                logger.warning(f"⚠️ {filename}: Issues found")
                for issue in file_results["issues"]:
                    logger.warning(f"   - {issue}")

        except Exception as e:
            results["issues"].append(f"❌ Error reading {filename}: {str(e)}")

    # Overall status
    total_issues = len(results["issues"]) + sum(
        len(f.get("issues", [])) for f in results["files_checked"].values()
    )

    if total_issues == 0:
        results["status"] = "good"
        logger.info("✅ LinkedIn 124k data format looks good!")
    else:
        results["status"] = "issues_found"
        logger.warning(f"⚠️ Found {total_issues} format issues")

    return results


def main():
    """Main format checking function."""

    print("🔍 Data Format Checker - Skills Demand Forecasting Platform")
    print("=" * 65)
    print()
    print("This tool quickly checks if your data files match expected formats")
    print("before running the full validation. Run this first to catch issues early.")
    print()

    # Check common data locations
    data_locations = {
        "LinkedIn 124k": "data/processed/linkedin_124k",
        "LinkedIn 1.3M": "data/processed/linkedin_1_3m"
    }

    all_results = {}

    for source, path in data_locations.items():
        print(f"📊 Checking {source} format...")
        print("-" * 40)

        if not Path(path).exists():
            print(f"❌ Directory not found: {path}")
            print("   Create this directory and add your CSV files")
            print()
            continue

        if source == "LinkedIn 124k":
            results = check_linkedin_124k_format(path)
            all_results[source] = results

        print()

    # Summary
    print("📋 FORMAT CHECK SUMMARY")
    print("=" * 50)

    for source, results in all_results.items():
        status_icon = "✅" if results["status"] == "good" else "⚠️"
        print(f"{status_icon} {source}: {results['status']}")

        if results["issues"]:
            print("   Issues to fix:")
            for issue in results["issues"]:
                print(f"   - {issue}")

    print()

    if all(r["status"] == "good" for r in all_results.values()):
        print("🎉 All data formats look good!")
        print("✅ Ready to run full validation: python scripts/validate_real_data.py")
    else:
        print("📝 Fix the format issues above, then run full validation.")
        print()
        print("💡 Common fixes:")
        print("   - Ensure CSV files have proper headers")
        print("   - Check that job_id columns are consistent across files")
        print("   - Verify salary columns contain numeric data")
        print("   - Make sure description columns aren't mostly empty")


if __name__ == "__main__":
    main()