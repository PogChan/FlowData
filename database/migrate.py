#!/usr/bin/env python3
"""
Database migration runner for options flow classifier.
Provides SQL migration files that can be executed manually in Supabase SQL editor.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List
from supabase import create_client, Client
import streamlit as st
from utils.config import config
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles database migrations for the options flow classifier."""

    def __init__(self, supabase_client: Client = None):
        self.supabase = supabase_client
        self.migrations_dir = Path(__file__).parent / "migrations"

    def get_migration_files(self) -> List[Path]:
        """Get all SQL migration files in order."""
        if not self.migrations_dir.exists():
            logger.error(f"Migrations directory not found: {self.migrations_dir}")
            return []

        sql_files = list(self.migrations_dir.glob("*.sql"))
        sql_files.sort()  # Sort by filename (001_, 002_, etc.)
        return sql_files

    def print_migration_instructions(self):
        """Print instructions for running migrations manually."""
        migration_files = self.get_migration_files()

        if not migration_files:
            logger.warning("No migration files found")
            return

        print("\n" + "="*80)
        print("DATABASE MIGRATION INSTRUCTIONS")
        print("="*80)
        print("\nTo set up the enhanced database schema, execute the following SQL files")
        print("in your Supabase SQL Editor in this order:\n")

        for i, migration_file in enumerate(migration_files, 1):
            print(f"{i}. {migration_file.name}")
            print(f"   Location: {migration_file}")
            print(f"   Description: {self._get_migration_description(migration_file)}")
            print()

        print("Steps to execute:")
        print("1. Go to your Supabase project dashboard")
        print("2. Navigate to SQL Editor")
        print("3. Copy and paste each migration file content in order")
        print("4. Execute each migration")
        print("\nAlternatively, you can run: python database/migrate.py --execute")
        print("(This requires proper Supabase connection setup)")
        print("="*80 + "\n")

    def _get_migration_description(self, migration_file: Path) -> str:
        """Extract description from migration file."""
        try:
            with open(migration_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('-- Migration:'):
                    return first_line[13:].strip()
        except:
            pass
        return "Database schema migration"

    def validate_schema(self) -> bool:
        """Validate that the required tables exist."""
        if not self.supabase:
            logger.error("No Supabase client available for validation")
            return False

        required_tables = ['options_flow', 'classification_rules', 'options_chain_cache']

        try:
            for table in required_tables:
                # Try to query the table to see if it exists
                result = self.supabase.table(table).select('*').limit(1).execute()
                logger.info(f"✓ Table '{table}' exists and is accessible")

            logger.info("All required tables are present")
            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def show_migration_status(self):
        """Show the current migration status."""
        migration_files = self.get_migration_files()

        print("\n" + "="*60)
        print("MIGRATION STATUS")
        print("="*60)

        for migration_file in migration_files:
            print(f"📄 {migration_file.name}")
            print(f"   {self._get_migration_description(migration_file)}")

        print(f"\nTotal migrations: {len(migration_files)}")

        if self.supabase:
            if self.validate_schema():
                print("✅ Schema validation: PASSED")
            else:
                print("❌ Schema validation: FAILED")
        else:
            print("⚠️  Schema validation: SKIPPED (no database connection)")

        print("="*60 + "\n")

def init_supabase_client() -> Client:
    """Initialize Supabase client using ConfigManager."""
    try:
        db = config.database
        return create_client(db.url, db.key)
    except Exception as e:
        logger.warning(f"Supabase configuration error: {e}")
        return None

def main():
    """Main migration runner."""
    try:
        logger.info("Database Migration Tool for Options Flow Classifier")

        # Initialize Supabase client (optional)
        supabase = None
        try:
            supabase = init_supabase_client()
            if supabase:
                logger.info("Connected to Supabase")
        except Exception as e:
            logger.warning(f"Could not connect to Supabase: {e}")

        # Create migrator
        migrator = DatabaseMigrator(supabase)

        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == '--status':
            migrator.show_migration_status()
        else:
            migrator.print_migration_instructions()

    except Exception as e:
        logger.error(f"Migration tool failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
