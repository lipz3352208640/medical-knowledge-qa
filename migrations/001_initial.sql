CREATE TABLE IF NOT EXISTS drug (
    id CHAR(24) PRIMARY KEY,
    raw_name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS document (
    id CHAR(24) PRIMARY KEY,
    drug_id CHAR(24) NOT NULL,
    source_file TEXT NOT NULL,
    file_hash CHAR(64) NOT NULL,
    parser_version VARCHAR(64) NOT NULL,
    page_count INT NOT NULL,
    parse_status ENUM('READY', 'REVIEW') NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY document_file_parser_uq (file_hash, parser_version),
    CONSTRAINT document_drug_fk FOREIGN KEY (drug_id) REFERENCES drug(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS section (
    id CHAR(24) PRIMARY KEY,
    document_id CHAR(24) NOT NULL,
    raw_name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    start_page INT NOT NULL,
    end_page INT NOT NULL,
    heading_page INT NOT NULL,
    heading_bbox JSON NOT NULL,
    confidence DOUBLE NOT NULL,
    CONSTRAINT section_document_fk FOREIGN KEY (document_id) REFERENCES document(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS element (
    id CHAR(24) PRIMARY KEY,
    document_id CHAR(24) NOT NULL,
    section_id CHAR(24) NOT NULL,
    page INT NOT NULL,
    bbox JSON NOT NULL,
    element_type VARCHAR(32) NOT NULL,
    raw_text LONGTEXT NOT NULL,
    normalized_text LONGTEXT NOT NULL,
    reading_order INT NOT NULL,
    confidence DOUBLE NOT NULL,
    values_json JSON NOT NULL,
    KEY element_source_idx (document_id, section_id, page),
    CONSTRAINT element_document_fk FOREIGN KEY (document_id) REFERENCES document(id),
    CONSTRAINT element_section_fk FOREIGN KEY (section_id) REFERENCES section(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS parsed_table (
    id CHAR(24) PRIMARY KEY,
    document_id CHAR(24) NOT NULL,
    section_id CHAR(24) NOT NULL,
    title TEXT NOT NULL,
    title_ref JSON,
    markdown LONGTEXT NOT NULL,
    raw_json JSON NOT NULL,
    normalized_json JSON NOT NULL,
    image_paths JSON NOT NULL,
    refs_json JSON NOT NULL,
    warnings_json JSON NOT NULL,
    CONSTRAINT parsed_table_document_fk FOREIGN KEY (document_id) REFERENCES document(id),
    CONSTRAINT parsed_table_section_fk FOREIGN KEY (section_id) REFERENCES section(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS table_cell (
    table_id CHAR(24) NOT NULL,
    row_index INT NOT NULL,
    column_index INT NOT NULL,
    document_id CHAR(24) NOT NULL,
    section_id CHAR(24) NOT NULL,
    page INT NOT NULL,
    bbox JSON NOT NULL,
    raw_value TEXT,
    normalized_value TEXT,
    inherited_from_row INT,
    PRIMARY KEY (table_id, row_index, column_index),
    CONSTRAINT table_cell_table_fk FOREIGN KEY (table_id) REFERENCES parsed_table(id),
    CONSTRAINT table_cell_document_fk FOREIGN KEY (document_id) REFERENCES document(id),
    CONSTRAINT table_cell_section_fk FOREIGN KEY (section_id) REFERENCES section(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS table_row (
    id CHAR(24) PRIMARY KEY,
    table_id CHAR(24) NOT NULL,
    document_id CHAR(24) NOT NULL,
    section_id CHAR(24) NOT NULL,
    row_index INT NOT NULL,
    page INT NOT NULL,
    bbox JSON NOT NULL,
    refs_json JSON NOT NULL,
    raw_cells JSON NOT NULL,
    normalized_cells JSON NOT NULL,
    retrieval_text LONGTEXT NOT NULL,
    values_json JSON NOT NULL,
    KEY table_row_source_idx (document_id, section_id, page),
    CONSTRAINT table_row_table_fk FOREIGN KEY (table_id) REFERENCES parsed_table(id),
    CONSTRAINT table_row_document_fk FOREIGN KEY (document_id) REFERENCES document(id),
    CONSTRAINT table_row_section_fk FOREIGN KEY (section_id) REFERENCES section(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS image (
    id CHAR(24) PRIMARY KEY,
    document_id CHAR(24) NOT NULL,
    section_id CHAR(24) NOT NULL,
    page INT NOT NULL,
    bbox JSON NOT NULL,
    image_type VARCHAR(64) NOT NULL,
    path TEXT NOT NULL,
    caption TEXT,
    confidence DOUBLE NOT NULL,
    KEY image_source_idx (document_id, section_id, page),
    CONSTRAINT image_document_fk FOREIGN KEY (document_id) REFERENCES document(id),
    CONSTRAINT image_section_fk FOREIGN KEY (section_id) REFERENCES section(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS parse_report (
    document_id CHAR(24) PRIMARY KEY,
    report_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT parse_report_document_fk FOREIGN KEY (document_id) REFERENCES document(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
