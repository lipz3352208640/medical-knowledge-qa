CREATE TABLE document_discard_log (
    document_id CHAR(24) PRIMARY KEY,
    actor VARCHAR(64) NOT NULL,
    reason VARCHAR(2000) NOT NULL,
    discarded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT document_discard_document_fk FOREIGN KEY (document_id) REFERENCES document(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
