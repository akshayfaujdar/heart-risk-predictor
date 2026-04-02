-- CardioAI MySQL initialisation script
-- This runs automatically when the MySQL Docker container starts for the first time.

CREATE DATABASE IF NOT EXISTS cardioai_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cardioai_db;

-- SQLAlchemy will create all tables via init_db() on FastAPI startup.
-- This file is kept here for manual setup or migration reference.

-- Verify connection
SELECT 'CardioAI MySQL database ready ✅' AS status;
