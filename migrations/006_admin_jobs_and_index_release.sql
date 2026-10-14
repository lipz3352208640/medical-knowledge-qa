CREATE TABLE IF NOT EXISTS rag_job (
    id CHAR(32) PRIMARY KEY,
    kind ENUM('REPROCESS','INDEX') NOT NULL,
    document_id CHAR(24) NOT NULL,
    source_file TEXT NOT NULL,
    status ENUM('PENDING','RUNNING','VALIDATING','READY','NEEDS_REVIEW','FAILED') NOT NULL,
    attempts INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    started_at TIMESTAMP(6) NULL,
    finished_at TIMESTAMP(6) NULL,
    result_json JSON NULL,
    error_summary TEXT NULL,
    KEY rag_job_claim_idx (status,created_at),
    KEY rag_job_document_idx (document_id,status),
    CONSTRAINT rag_job_document_fk FOREIGN KEY (document_id) REFERENCES document(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS index_release (
    version VARCHAR(128) PRIMARY KEY,
    status ENUM('RUNNING','READY','FAILED') NOT NULL,
    collection_name VARCHAR(255) NOT NULL,
    document_count INT NOT NULL DEFAULT 0,
    failed_count INT NOT NULL DEFAULT 0,
    started_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    completed_at TIMESTAMP(6) NULL,
    error_summary TEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
