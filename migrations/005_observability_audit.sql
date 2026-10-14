CREATE TABLE IF NOT EXISTS qa_audit (
    answer_id CHAR(32) PRIMARY KEY COMMENT '问答审计唯一ID',
    trace_id CHAR(32) NULL COMMENT '可选Trace ID',
    request_id CHAR(64) NOT NULL COMMENT '请求关联ID',
    occurred_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    status ENUM('STARTED','COMPLETED','FAILED','INCOMPLETE') NOT NULL,
    user_pseudonym VARCHAR(128) NULL,
    question_digest CHAR(64) NOT NULL,
    question_ciphertext LONGTEXT NULL,
    answer_digest CHAR(64) NULL,
    answer_ciphertext LONGTEXT NULL,
    cited_evidence JSON NOT NULL,
    rule_version VARCHAR(128) NULL,
    index_version VARCHAR(128) NULL,
    prompt_version VARCHAR(128) NULL,
    model_id VARCHAR(255) NULL,
    no_evidence_reason TEXT NULL,
    citation_check_status VARCHAR(32) NULL,
    completed_at TIMESTAMP(6) NULL,
    failure_code VARCHAR(128) NULL,
    KEY qa_audit_occurred_idx (occurred_at),
    KEY qa_audit_trace_idx (trace_id),
    KEY qa_audit_status_idx (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='关键问答可靠审计记录';

CREATE TABLE IF NOT EXISTS parse_run (
    id CHAR(24) PRIMARY KEY,
    document_id CHAR(24) NOT NULL,
    document_version VARCHAR(128) NOT NULL,
    file_hash CHAR(64) NOT NULL,
    parser_version VARCHAR(64) NOT NULL,
    config_version VARCHAR(128) NULL,
    status ENUM('PENDING','RUNNING','VALIDATING','READY','NEEDS_REVIEW','FAILED') NOT NULL,
    started_at TIMESTAMP(6) NULL,
    finished_at TIMESTAMP(6) NULL,
    error_summary TEXT NULL,
    UNIQUE KEY parse_run_document_version_uq (document_id, parser_version),
    CONSTRAINT parse_run_document_fk FOREIGN KEY (document_id) REFERENCES document(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='不可变解析运行记录';

CREATE TABLE IF NOT EXISTS page_report (
    parse_run_id CHAR(24) NOT NULL,
    page INT NOT NULL,
    width DOUBLE NULL,
    height DOUBLE NULL,
    text_count INT NOT NULL DEFAULT 0,
    table_count INT NOT NULL DEFAULT 0,
    image_count INT NOT NULL DEFAULT 0,
    warnings_json JSON NOT NULL,
    PRIMARY KEY (parse_run_id, page),
    CONSTRAINT page_report_run_fk FOREIGN KEY (parse_run_id) REFERENCES parse_run(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='逐页解析质量报告';

CREATE TABLE IF NOT EXISTS element_report (
    id CHAR(24) PRIMARY KEY,
    parse_run_id CHAR(24) NOT NULL,
    element_id CHAR(24) NOT NULL,
    page INT NOT NULL,
    bbox JSON NOT NULL,
    bbox_coordinate_system VARCHAR(64) NOT NULL DEFAULT 'pdf_points_top_left',
    reading_order INT NOT NULL,
    element_type VARCHAR(64) NOT NULL,
    raw_text LONGTEXT NOT NULL,
    normalized_text LONGTEXT NOT NULL,
    section_path JSON NOT NULL,
    confidence DOUBLE NOT NULL,
    source_asset_id CHAR(24) NULL,
    table_id CHAR(24) NULL,
    parent_id CHAR(24) NULL,
    warning_codes JSON NOT NULL,
    KEY element_report_run_page_idx (parse_run_id, page),
    CONSTRAINT element_report_run_fk FOREIGN KEY (parse_run_id) REFERENCES parse_run(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='逐元素解析审计报告';
