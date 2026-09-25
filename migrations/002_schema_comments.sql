ALTER TABLE `drug` COMMENT = '药品主数据：保存说明书中的原始药品名及其规范化名称';
ALTER TABLE `drug`
    MODIFY COLUMN `id` CHAR(24) NOT NULL COMMENT '稳定药品ID，由规范化药品名生成',
    MODIFY COLUMN `raw_name` VARCHAR(255) NOT NULL COMMENT '从说明书提取的原始药品名称',
    MODIFY COLUMN `normalized_name` VARCHAR(255) NOT NULL COMMENT '规范化药品名称，用于关联和去重';

ALTER TABLE `document` COMMENT = '说明书文档版本：记录来源文件、文件哈希、解析器版本和解析状态';
ALTER TABLE `document`
    MODIFY COLUMN `id` CHAR(24) NOT NULL COMMENT '文档版本ID，由文件哈希和解析器版本生成',
    MODIFY COLUMN `drug_id` CHAR(24) NOT NULL COMMENT '所属药品ID，关联drug.id',
    MODIFY COLUMN `source_file` TEXT NOT NULL COMMENT '原始PDF文件路径',
    MODIFY COLUMN `file_hash` CHAR(64) NOT NULL COMMENT '原始PDF文件的SHA-256哈希',
    MODIFY COLUMN `parser_version` VARCHAR(64) NOT NULL COMMENT '生成本次解析结果的解析器版本',
    MODIFY COLUMN `page_count` INT NOT NULL COMMENT 'PDF总页数',
    MODIFY COLUMN `parse_status` ENUM('READY', 'REVIEW') NOT NULL COMMENT '解析状态：READY可用，REVIEW需要人工复核',
    MODIFY COLUMN `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '文档记录创建时间';

ALTER TABLE `section` COMMENT = '文档章节：保存章节原名、规范名、页码范围和标题位置';
ALTER TABLE `section`
    MODIFY COLUMN `id` CHAR(24) NOT NULL COMMENT '章节ID',
    MODIFY COLUMN `document_id` CHAR(24) NOT NULL COMMENT '所属文档版本ID，关联document.id',
    MODIFY COLUMN `raw_name` VARCHAR(255) NOT NULL COMMENT '说明书中的原始章节名称',
    MODIFY COLUMN `normalized_name` VARCHAR(255) NOT NULL COMMENT '映射后的标准章节名称',
    MODIFY COLUMN `start_page` INT NOT NULL COMMENT '章节起始页码，从1开始',
    MODIFY COLUMN `end_page` INT NOT NULL COMMENT '章节结束页码，从1开始',
    MODIFY COLUMN `heading_page` INT NOT NULL COMMENT '章节标题所在页码',
    MODIFY COLUMN `heading_bbox` JSON NOT NULL COMMENT '章节标题坐标[x0,y0,x1,y1]，PDF点，左上角为原点',
    MODIFY COLUMN `confidence` DOUBLE NOT NULL COMMENT '章节识别置信度，范围0到1';

ALTER TABLE `element` COMMENT = '页面文本元素：保存正文、列表、标题、页眉页脚等原始解析单元';
ALTER TABLE `element`
    MODIFY COLUMN `id` CHAR(24) NOT NULL COMMENT '页面元素ID',
    MODIFY COLUMN `document_id` CHAR(24) NOT NULL COMMENT '所属文档版本ID，关联document.id',
    MODIFY COLUMN `section_id` CHAR(24) NOT NULL COMMENT '所属章节ID，关联section.id',
    MODIFY COLUMN `page` INT NOT NULL COMMENT '元素所在页码，从1开始',
    MODIFY COLUMN `bbox` JSON NOT NULL COMMENT '元素坐标[x0,y0,x1,y1]，PDF点，左上角为原点',
    MODIFY COLUMN `element_type` VARCHAR(32) NOT NULL COMMENT '元素类型，如paragraph、list_item、section_heading',
    MODIFY COLUMN `raw_text` LONGTEXT NOT NULL COMMENT 'PDF中提取的原始文本',
    MODIFY COLUMN `normalized_text` LONGTEXT NOT NULL COMMENT '用于后续处理的规范化文本',
    MODIFY COLUMN `reading_order` INT NOT NULL COMMENT '元素在当前页中的阅读顺序',
    MODIFY COLUMN `confidence` DOUBLE NOT NULL COMMENT '元素识别置信度，范围0到1',
    MODIFY COLUMN `values_json` JSON NOT NULL COMMENT '文本中的剂量、单位、等级等原值与规范值';

ALTER TABLE `parsed_table` COMMENT = '结构化表格：保存表题、Markdown、原始JSON、规范JSON、裁剪图及跨页来源';
ALTER TABLE `parsed_table`
    MODIFY COLUMN `id` CHAR(24) NOT NULL COMMENT '结构化表格ID',
    MODIFY COLUMN `document_id` CHAR(24) NOT NULL COMMENT '所属文档版本ID，关联document.id',
    MODIFY COLUMN `section_id` CHAR(24) NOT NULL COMMENT '所属章节ID，关联section.id',
    MODIFY COLUMN `title` TEXT NOT NULL COMMENT '识别出的表格标题，可跨页回溯',
    MODIFY COLUMN `title_ref` JSON NULL COMMENT '表格标题的document、section、page和bbox来源',
    MODIFY COLUMN `markdown` LONGTEXT NOT NULL COMMENT '规范化表格的Markdown表示',
    MODIFY COLUMN `raw_json` JSON NOT NULL COMMENT '未清洗的原始行列数据和分页片段信息',
    MODIFY COLUMN `normalized_json` JSON NOT NULL COMMENT '合并单元格展开后的规范化数据及单元格来源',
    MODIFY COLUMN `image_paths` JSON NOT NULL COMMENT '各分页表格原始裁剪图路径列表',
    MODIFY COLUMN `refs_json` JSON NOT NULL COMMENT '表格各分页片段的document、section、page和bbox来源',
    MODIFY COLUMN `warnings_json` JSON NOT NULL COMMENT '跨页拼接等需要人工核查的警告';

ALTER TABLE `table_cell` COMMENT = '表格单元格：逐格保存原值、规范值、坐标和合并单元格继承关系';
ALTER TABLE `table_cell`
    MODIFY COLUMN `table_id` CHAR(24) NOT NULL COMMENT '所属结构化表格ID，关联parsed_table.id',
    MODIFY COLUMN `row_index` INT NOT NULL COMMENT '单元格行号，从0开始，0通常为表头',
    MODIFY COLUMN `column_index` INT NOT NULL COMMENT '单元格列号，从0开始',
    MODIFY COLUMN `document_id` CHAR(24) NOT NULL COMMENT '所属文档版本ID，关联document.id',
    MODIFY COLUMN `section_id` CHAR(24) NOT NULL COMMENT '所属章节ID，关联section.id',
    MODIFY COLUMN `page` INT NOT NULL COMMENT '单元格所在页码，从1开始',
    MODIFY COLUMN `bbox` JSON NOT NULL COMMENT '单元格坐标[x0,y0,x1,y1]，PDF点，左上角为原点',
    MODIFY COLUMN `raw_value` TEXT NULL COMMENT 'PDF提取的原始单元格值，空白合并格可为NULL',
    MODIFY COLUMN `normalized_value` TEXT NULL COMMENT '清洗及受控继承后的规范化单元格值',
    MODIFY COLUMN `inherited_from_row` INT NULL COMMENT '合并单元格规范值继承自哪一行，未继承时为NULL';

ALTER TABLE `table_row` COMMENT = '表格行检索单元：将每个数据行保存为可追溯的结构和完整语义文本';
ALTER TABLE `table_row`
    MODIFY COLUMN `id` CHAR(24) NOT NULL COMMENT '表格行ID',
    MODIFY COLUMN `table_id` CHAR(24) NOT NULL COMMENT '所属结构化表格ID，关联parsed_table.id',
    MODIFY COLUMN `document_id` CHAR(24) NOT NULL COMMENT '所属文档版本ID，关联document.id',
    MODIFY COLUMN `section_id` CHAR(24) NOT NULL COMMENT '所属章节ID，关联section.id',
    MODIFY COLUMN `row_index` INT NOT NULL COMMENT '数据行在表格中的行号，从1开始，0为表头',
    MODIFY COLUMN `page` INT NOT NULL COMMENT '数据行所在页码，从1开始',
    MODIFY COLUMN `bbox` JSON NOT NULL COMMENT '数据行坐标[x0,y0,x1,y1]，PDF点，左上角为原点',
    MODIFY COLUMN `refs_json` JSON NOT NULL COMMENT '该行全部分页来源的document、section、page和bbox',
    MODIFY COLUMN `raw_cells` JSON NOT NULL COMMENT '该行各单元格的原始值数组',
    MODIFY COLUMN `normalized_cells` JSON NOT NULL COMMENT '该行各单元格的规范化值数组',
    MODIFY COLUMN `retrieval_text` LONGTEXT NOT NULL COMMENT '包含药品名、表头和行值的完整语义文本',
    MODIFY COLUMN `values_json` JSON NOT NULL COMMENT '行内容中的剂量、单位、等级等原值与规范值';

ALTER TABLE `image` COMMENT = '图片资产：保存化学结构图、图表及其他内嵌图片的文件和页面来源';
ALTER TABLE `image`
    MODIFY COLUMN `id` CHAR(24) NOT NULL COMMENT '图片资产ID',
    MODIFY COLUMN `document_id` CHAR(24) NOT NULL COMMENT '所属文档版本ID，关联document.id',
    MODIFY COLUMN `section_id` CHAR(24) NOT NULL COMMENT '所属章节ID，关联section.id',
    MODIFY COLUMN `page` INT NOT NULL COMMENT '图片所在页码，从1开始',
    MODIFY COLUMN `bbox` JSON NOT NULL COMMENT '图片坐标[x0,y0,x1,y1]，PDF点，左上角为原点',
    MODIFY COLUMN `image_type` VARCHAR(64) NOT NULL COMMENT '图片类型，如chemical_structure、chart、glyph_fragment',
    MODIFY COLUMN `path` TEXT NOT NULL COMMENT '提取或裁剪后的图片文件路径',
    MODIFY COLUMN `caption` TEXT NULL COMMENT '识别到的图片标题或说明文字',
    MODIFY COLUMN `confidence` DOUBLE NOT NULL COMMENT '图片类型识别置信度，范围0到1';

ALTER TABLE `parse_report` COMMENT = '解析质量报告：保存页面统计、失败页、空白页、警告和待复核状态';
ALTER TABLE `parse_report`
    MODIFY COLUMN `document_id` CHAR(24) NOT NULL COMMENT '文档版本ID，同时作为主键并关联document.id',
    MODIFY COLUMN `report_json` JSON NOT NULL COMMENT '完整机器可读解析质量报告',
    MODIFY COLUMN `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '报告创建时间';
