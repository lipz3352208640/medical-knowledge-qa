CREATE TABLE IF NOT EXISTS `chunk_source_segment` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `chunk_id` CHAR(24) NOT NULL,
    `parse_run_id` CHAR(24) NULL,
    `index_version` CHAR(64) NULL,
    `ordinal` INT NOT NULL,
    `element_id` CHAR(24) NULL,
    `source_id` CHAR(24) NULL,
    `source_kind` VARCHAR(32) NOT NULL,
    `page_number` INT NULL,
    `bbox` JSON NULL,
    `element_start` INT NULL,
    `element_end` INT NULL,
    `chunk_start` INT NULL,
    `chunk_end` INT NULL,
    `location_status` ENUM('LOCATABLE','UNLOCATABLE') NOT NULL,
    `created_at` TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY `source_chunk_order_idx` (`chunk_id`, `ordinal`),
    KEY `source_run_element_idx` (`parse_run_id`, `element_id`),
    CONSTRAINT `source_chunk_fk` FOREIGN KEY (`chunk_id`) REFERENCES `chunk` (`id`) ON DELETE CASCADE,
    CONSTRAINT `source_run_fk` FOREIGN KEY (`parse_run_id`) REFERENCES `parse_run` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Existing refs and source ID lists cannot establish exact character ranges.
-- Keep one explicit marker per legacy chunk until a real source rebuild runs.
INSERT INTO `chunk_source_segment`
    (`chunk_id`,`parse_run_id`,`index_version`,`ordinal`,`source_kind`,`location_status`)
SELECT c.id, NULL, dis.chunk_manifest_hash, 0, 'legacy', 'UNLOCATABLE'
FROM `chunk` c
LEFT JOIN `document_index_state` dis ON dis.document_id=c.document_id;
