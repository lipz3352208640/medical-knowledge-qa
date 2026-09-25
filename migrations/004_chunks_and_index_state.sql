CREATE TABLE IF NOT EXISTS `chunk` (
    `id` CHAR(24) PRIMARY KEY COMMENT '稳定分块ID',
    `document_id` CHAR(24) NOT NULL COMMENT '来源文档版本ID',
    `section_id` CHAR(24) NOT NULL COMMENT '来源章节ID',
    `parent_id` CHAR(24) NULL COMMENT '父级章节或整表chunk ID',
    `content` LONGTEXT NOT NULL COMMENT '忠于原文的分块正文',
    `retrieval_text` LONGTEXT NOT NULL COMMENT '补全药品、章节和表名语境后的检索文本',
    `content_type` VARCHAR(32) NOT NULL COMMENT 'section、paragraph、list_item、table、table_row或image',
    `is_parent` BOOLEAN NOT NULL COMMENT '是否为只用于上下文恢复的父级chunk',
    `token_count` INT NOT NULL COMMENT '不依赖模型tokenizer的确定性近似token数',
    `page_start` INT NOT NULL COMMENT '起始页码，从1开始',
    `page_end` INT NOT NULL COMMENT '结束页码，从1开始',
    `bbox` JSON NULL COMMENT '单页chunk的合并坐标；跨页时为NULL',
    `reading_order` INT NOT NULL COMMENT '跨页可排序的阅读顺序',
    `refs_json` JSON NOT NULL COMMENT '全部document、section、page和bbox来源',
    `metadata_json` JSON NOT NULL COMMENT '药品、章节、人群、意图和版本等过滤元数据',
    `content_hash` CHAR(64) NOT NULL COMMENT '原文内容SHA-256',
    `chunker_version` VARCHAR(64) NOT NULL COMMENT '分块算法版本',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY `chunk_document_idx` (`document_id`, `is_parent`),
    KEY `chunk_section_idx` (`section_id`, `reading_order`),
    KEY `chunk_parent_idx` (`parent_id`),
    KEY `chunk_type_idx` (`content_type`),
    FULLTEXT KEY `chunk_retrieval_ft` (`retrieval_text`) WITH PARSER ngram,
    CONSTRAINT `chunk_document_fk` FOREIGN KEY (`document_id`) REFERENCES `document` (`id`),
    CONSTRAINT `chunk_section_fk` FOREIGN KEY (`section_id`) REFERENCES `section` (`id`),
    CONSTRAINT `chunk_parent_fk` FOREIGN KEY (`parent_id`) REFERENCES `chunk` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='父子检索分块：child用于召回，parent用于恢复完整上下文';

CREATE TABLE IF NOT EXISTS `document_index_state` (
    `document_id` CHAR(24) PRIMARY KEY COMMENT '文档版本ID',
    `chunker_version` VARCHAR(64) NOT NULL COMMENT '当前chunk使用的分块算法版本',
    `chunk_manifest_hash` CHAR(64) NOT NULL COMMENT '全部chunk ID与内容hash组成的清单hash',
    `vector_status` ENUM('PENDING', 'INDEXING', 'READY', 'FAILED', 'STALE') NOT NULL DEFAULT 'PENDING',
    `embedding_model` VARCHAR(255) NULL COMMENT '生成当前向量的模型标签',
    `collection_name` VARCHAR(255) NULL COMMENT 'Qdrant collection名称',
    `point_count` INT NOT NULL DEFAULT 0 COMMENT '成功写入的child向量数量',
    `last_error` TEXT NULL COMMENT '最近一次索引失败原因',
    `indexed_at` TIMESTAMP NULL COMMENT '最近成功完成向量索引的时间',
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT `document_index_state_document_fk`
        FOREIGN KEY (`document_id`) REFERENCES `document` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='MySQL chunk与外部Qdrant向量索引之间的可恢复状态';
