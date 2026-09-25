ALTER TABLE `drug`
    CHANGE COLUMN `raw_name` `generic_name_raw` VARCHAR(255) NOT NULL COMMENT '说明书中的原始通用名称',
    CHANGE COLUMN `normalized_name` `generic_name_normalized` VARCHAR(255) NOT NULL COMMENT '规范化通用名称，作为药品实体的稳定归并键';
ALTER TABLE `drug` COMMENT = '规范药品实体：以规范化通用名称归并商品名、英文名和其他别名';

CREATE TABLE IF NOT EXISTS `drug_product` (
    `id` CHAR(24) PRIMARY KEY COMMENT '药品商品ID',
    `drug_id` CHAR(24) NOT NULL COMMENT '所属规范药品ID，关联drug.id',
    `identity_hash` CHAR(64) NOT NULL COMMENT '商品身份哈希，优先由批准文号集合生成',
    `identity_source` ENUM('APPROVAL_NUMBER', 'GENERIC_BRAND', 'LEGACY') NOT NULL COMMENT '商品身份判定依据',
    `brand_name_raw` VARCHAR(255) NULL COMMENT '说明书中的原始商品名',
    `brand_name_normalized` VARCHAR(255) NULL COMMENT '规范化商品名',
    `english_name_raw` VARCHAR(255) NULL COMMENT '说明书中的英文名称',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '商品记录创建时间',
    UNIQUE KEY `drug_product_identity_uq` (`identity_hash`),
    KEY `drug_product_drug_idx` (`drug_id`),
    CONSTRAINT `drug_product_drug_fk` FOREIGN KEY (`drug_id`) REFERENCES `drug` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='上市药品商品：区分同一通用药品的不同品牌或批准文号集合';

CREATE TABLE IF NOT EXISTS `product_approval_number` (
    `product_id` CHAR(24) NOT NULL COMMENT '所属药品商品ID，关联drug_product.id',
    `approval_number` VARCHAR(64) NOT NULL COMMENT '规范化批准文号',
    PRIMARY KEY (`product_id`, `approval_number`),
    UNIQUE KEY `product_approval_number_uq` (`approval_number`),
    CONSTRAINT `product_approval_product_fk` FOREIGN KEY (`product_id`) REFERENCES `drug_product` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='药品商品批准文号：支持一份说明书覆盖多个规格和批准文号';

ALTER TABLE `document`
    ADD COLUMN `product_id` CHAR(24) NULL AFTER `id`,
    ADD COLUMN `revision_date` DATE NULL AFTER `parser_version`,
    ADD COLUMN `version_status` ENUM('ACTIVE', 'SUPERSEDED', 'WITHDRAWN', 'PENDING') NOT NULL DEFAULT 'SUPERSEDED' AFTER `parse_status`,
    ADD COLUMN `supersedes_document_id` CHAR(24) NULL AFTER `version_status`,
    ADD COLUMN `effective_from` TIMESTAMP NULL AFTER `supersedes_document_id`,
    ADD COLUMN `effective_to` TIMESTAMP NULL AFTER `effective_from`;

INSERT IGNORE INTO `drug_product`
    (`id`, `drug_id`, `identity_hash`, `identity_source`, `brand_name_raw`, `brand_name_normalized`, `english_name_raw`)
SELECT SUBSTRING(SHA2(CONCAT('legacy-product:', `id`), 256), 1, 24),
       `id`, SHA2(CONCAT('legacy-product:', `id`), 256), 'LEGACY', NULL, NULL, NULL
FROM `drug`;

UPDATE `document` AS d
JOIN `drug_product` AS p ON p.`drug_id` = d.`drug_id` AND p.`identity_source` = 'LEGACY'
SET d.`product_id` = p.`id`
WHERE d.`product_id` IS NULL;

UPDATE `document` AS d
JOIN (
    SELECT `id`, ROW_NUMBER() OVER (
        PARTITION BY `product_id` ORDER BY `created_at` DESC, `id` DESC
    ) AS row_number_in_product
    FROM `document`
) AS ranked ON ranked.`id` = d.`id`
SET d.`version_status` = IF(ranked.`row_number_in_product` = 1, 'ACTIVE', 'SUPERSEDED'),
    d.`effective_from` = IF(ranked.`row_number_in_product` = 1, d.`created_at`, d.`effective_from`);

ALTER TABLE `document`
    MODIFY COLUMN `product_id` CHAR(24) NOT NULL COMMENT '所属药品商品ID，关联drug_product.id',
    MODIFY COLUMN `revision_date` DATE NULL COMMENT '说明书最新修改或修订日期',
    MODIFY COLUMN `version_status` ENUM('ACTIVE', 'SUPERSEDED', 'WITHDRAWN', 'PENDING') NOT NULL COMMENT '业务版本状态：当前有效、已替代、已撤回或待确认',
    MODIFY COLUMN `supersedes_document_id` CHAR(24) NULL COMMENT '本版本替代的上一文档版本ID',
    MODIFY COLUMN `effective_from` TIMESTAMP NULL COMMENT '成为当前有效版本的时间',
    MODIFY COLUMN `effective_to` TIMESTAMP NULL COMMENT '停止作为当前有效版本的时间',
    ADD COLUMN `active_product_id` CHAR(24)
        GENERATED ALWAYS AS (CASE WHEN `version_status` = 'ACTIVE' THEN `product_id` ELSE NULL END) STORED
        COMMENT '激活版本唯一约束辅助列，业务代码不直接写入',
    ADD UNIQUE KEY `document_one_active_product_uq` (`active_product_id`),
    ADD KEY `document_product_idx` (`product_id`),
    ADD KEY `document_supersedes_idx` (`supersedes_document_id`),
    ADD CONSTRAINT `document_product_fk` FOREIGN KEY (`product_id`) REFERENCES `drug_product` (`id`),
    ADD CONSTRAINT `document_supersedes_fk` FOREIGN KEY (`supersedes_document_id`) REFERENCES `document` (`id`),
    DROP FOREIGN KEY `document_drug_fk`,
    DROP COLUMN `drug_id`;
ALTER TABLE `document` COMMENT = '说明书文档版本：属于一个药品商品，并明确当前有效、历史、撤回或待确认状态';

CREATE TABLE IF NOT EXISTS `drug_alias` (
    `id` CHAR(24) PRIMARY KEY COMMENT '名称记录ID',
    `drug_id` CHAR(24) NOT NULL COMMENT '所属规范药品ID，关联drug.id',
    `product_id` CHAR(24) NULL COMMENT '商品名所属商品ID；通用别名可为空',
    `source_document_id` CHAR(24) NULL COMMENT '该名称首次或本次取自的说明书文档ID',
    `alias_type` ENUM('GENERIC', 'BRAND', 'ENGLISH', 'OTHER') NOT NULL COMMENT '名称类型：通用名、商品名、英文名或其他别名',
    `raw_name` VARCHAR(255) NOT NULL COMMENT '说明书中的原始名称',
    `normalized_name` VARCHAR(255) NOT NULL COMMENT '用于匹配的规范化名称',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '名称记录创建时间',
    UNIQUE KEY `drug_alias_scope_uq` (`drug_id`, `product_id`, `alias_type`, `normalized_name`),
    KEY `drug_alias_normalized_idx` (`normalized_name`),
    CONSTRAINT `drug_alias_drug_fk` FOREIGN KEY (`drug_id`) REFERENCES `drug` (`id`),
    CONSTRAINT `drug_alias_product_fk` FOREIGN KEY (`product_id`) REFERENCES `drug_product` (`id`),
    CONSTRAINT `drug_alias_document_fk` FOREIGN KEY (`source_document_id`) REFERENCES `document` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='药品名称与别名：统一保存通用名、商品名、英文名及其他别名并保留来源';
