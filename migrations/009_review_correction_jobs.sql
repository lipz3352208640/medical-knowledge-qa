ALTER TABLE `rag_job` MODIFY `kind` ENUM('IMPORT','REPROCESS','INDEX','CORRECTION') NOT NULL;

ALTER TABLE `element` ADD COLUMN `is_bold` BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS `review_correction_batch` (
    `batch_key` CHAR(64) PRIMARY KEY,
    `job_id` CHAR(32) NOT NULL UNIQUE,
    `source_document_id` CHAR(24) NOT NULL,
    `source_parse_run_id` CHAR(24) NOT NULL,
    `source_index_version` CHAR(64) NOT NULL,
    `source_file_hash` CHAR(64) NOT NULL,
    `proposal_manifest_json` JSON NOT NULL,
    `target_document_id` CHAR(24) NULL,
    `created_at` TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `finished_at` TIMESTAMP(6) NULL,
    CONSTRAINT `review_batch_job_fk` FOREIGN KEY (`job_id`) REFERENCES `rag_job` (`id`),
    CONSTRAINT `review_batch_source_fk` FOREIGN KEY (`source_document_id`) REFERENCES `document` (`id`),
    CONSTRAINT `review_batch_run_fk` FOREIGN KEY (`source_parse_run_id`) REFERENCES `parse_run` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
