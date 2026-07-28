"""药品数据库 MCP Server — SQLite 本地药典

工具:
- get_drug_info(drug_name): 查询药品详细信息
- check_drug_interaction(drug_a, drug_b): 检查两种药品的相互作用

首次使用自动创建 SQLite 数据库并插入约 50 条常用药品示例数据。
"""

import os
import logging
import aiosqlite
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# 数据库文件路径（与 server 模块同目录）
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drug_data.db")

# ── 工具定义 ────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_drug_info",
        "description": "查询药品详细信息，包括适应症、副作用、禁忌、用法用量等",
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "药品名称（中文），如'阿司匹林'",
                }
            },
            "required": ["drug_name"],
        },
    },
    {
        "name": "check_drug_interaction",
        "description": "检查两种药品之间是否存在相互作用及风险等级",
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_a": {"type": "string", "description": "第一种药品名称"},
                "drug_b": {"type": "string", "description": "第二种药品名称"},
            },
            "required": ["drug_a", "drug_b"],
        },
    },
]

# ── 种子数据 ────────────────────────────────────────────
# 50 种常用药品 (name, category, indications, side_effects, contraindications, dosage, notes)

DRUGS_SEED = [
    # ── 解热镇痛药 ──
    ("阿司匹林", "解热镇痛药",
     "用于缓解轻度至中度疼痛（头痛、牙痛、肌肉痛），退热，抗血小板聚集预防心脑血管疾病",
     "胃肠道刺激、恶心、胃溃疡、出血时间延长、耳鸣（大剂量）",
     "活动性胃溃疡、出血体质、对阿司匹林过敏、妊娠晚期、严重肝肾功能不全",
     "解热镇痛：300-600mg/次；抗血小板：75-100mg/日",
     "与布洛芬等NSAIDs联用增加胃肠道出血风险"),

    ("布洛芬", "解热镇痛药",
     "缓解各种疼痛（头痛、关节痛、肌肉痛、牙痛、痛经），退热",
     "胃肠道不适、头晕、皮疹、肾功能损害（长期大剂量）",
     "活动性消化性溃疡、严重肝肾功能不全、对阿司匹林过敏者",
     "成人：200-400mg/次，每日不超过1.2g",
     "饭后服用可减轻胃肠道刺激"),

    ("对乙酰氨基酚", "解热镇痛药",
     "退热，缓解轻中度疼痛（头痛、肌肉痛、关节痛）",
     "常规剂量不良反应少，过量可致严重肝损伤",
     "严重肝病、对本品过敏者",
     "成人：500-1000mg/次，每日不超过4g",
     "为妊娠期首选的解热镇痛药"),

    ("双氯芬酸", "解热镇痛药",
     "缓解各种关节炎、软组织损伤、术后疼痛及痛风急性发作",
     "胃肠道反应、头痛、头晕、肝功能异常",
     "消化道溃疡、严重心衰、对NSAIDs过敏",
     "成人：75-150mg/日，分次服用",
     "外用凝胶剂型安全性更高"),

    ("萘普生", "解热镇痛药",
     "类风湿关节炎、骨关节炎、强直性脊柱炎、痛风、痛经",
     "胃肠道不适、头痛、嗜睡、耳鸣",
     "对阿司匹林过敏者、活动性溃疡",
     "成人：250-500mg/次，每日2次",
     "半衰期较长，每日服药次数少"),

    ("塞来昔布", "解热镇痛药",
     "骨关节炎、类风湿关节炎、强直性脊柱炎",
     "胃肠道反应较轻，增加心血管风险",
     "对磺胺类过敏、严重冠心病",
     "成人：100-200mg/次，每日1-2次",
     "选择性COX-2抑制剂，胃肠道安全性较好"),

    ("美洛昔康", "解热镇痛药",
     "类风湿关节炎、骨关节炎",
     "胃肠道反应、头晕、皮疹",
     "活动性溃疡、严重肝肾功能不全",
     "成人：7.5-15mg/日",
     "选择性COX-2抑制剂"),

    # ── 抗生素 ──
    ("阿莫西林", "抗生素",
     "敏感菌引起的呼吸道感染、泌尿道感染、皮肤软组织感染",
     "皮疹、腹泻、恶心，偶见过敏性休克",
     "青霉素过敏者禁用",
     "成人：500mg/次，每日3次",
     "用前需询问青霉素过敏史"),

    ("头孢克肟", "抗生素",
     "呼吸道感染、泌尿道感染、胆道感染、中耳炎",
     "腹泻、恶心、皮疹、肝功能异常",
     "头孢类过敏者禁用",
     "成人：100mg/次，每日2次",
     "第三代头孢菌素"),

    ("左氧氟沙星", "抗生素",
     "呼吸道感染、泌尿道感染、肠道感染、皮肤软组织感染",
     "胃肠道反应、头晕、失眠，QT间期延长",
     "未成年人、孕妇及哺乳期、癫痫患者",
     "成人：500mg/日，分1-2次",
     "喹诺酮类，18岁以下禁用"),

    ("阿奇霉素", "抗生素",
     "呼吸道感染、皮肤软组织感染、泌尿生殖道感染",
     "胃肠道反应、头痛、肝功能异常",
     "对大环内酯类过敏、严重肝病",
     "成人：500mg/日，连用3天",
     "大环内酯类，半衰期长"),

    ("克拉霉素", "抗生素",
     "呼吸道感染、皮肤软组织感染、幽门螺杆菌根除",
     "胃肠道反应、味觉异常、头痛",
     "对大环内酯类过敏",
     "成人：250-500mg/次，每日2次",
     "常用于根除幽门螺杆菌三联疗法"),

    ("甲硝唑", "抗生素",
     "厌氧菌感染、滴虫感染、阿米巴病、幽门螺杆菌根除",
     "金属味觉、胃肠道反应、头痛",
     "妊娠早期、哺乳期、有血液病史",
     "成人：200-400mg/次，每日3次",
     "服药期间及停药后禁酒"),

    ("头孢拉定", "抗生素",
     "呼吸道感染、泌尿道感染、皮肤软组织感染",
     "腹泻、恶心、皮疹",
     "头孢类过敏者禁用",
     "成人：250-500mg/次，每日4次",
     "第一代头孢菌素，口服吸收好"),

    ("青霉素V", "抗生素",
     "链球菌咽炎、轻度呼吸道感染、预防风湿热复发",
     "皮疹、腹泻，过敏反应",
     "青霉素过敏者禁用",
     "成人：250-500mg/次，每日3-4次",
     "耐酸口服青霉素"),

    # ── 降压药 ──
    ("硝苯地平", "降压药",
     "高血压、冠心病心绞痛（特别是变异型心绞痛）",
     "头痛、面红、下肢水肿、心悸、头晕",
     "严重低血压、心源性休克",
     "控释片：30-60mg/日",
     "二氢吡啶类钙通道阻滞剂"),

    ("氨氯地平", "降压药",
     "高血压、稳定型心绞痛",
     "头痛、水肿、疲劳、恶心、面红",
     "严重低血压、对二氢吡啶类过敏",
     "成人：5-10mg/日",
     "长效钙通道阻滞剂，每日一次"),

    ("卡托普利", "降压药",
     "高血压、心力衰竭、心肌梗死后",
     "干咳、皮疹、味觉异常、高钾血症",
     "妊娠期、双侧肾动脉狭窄、血管神经性水肿史",
     "成人：25-50mg/次，每日2-3次",
     "ACEI类药物，可能引起干咳"),

    ("缬沙坦", "降压药",
     "高血压、心力衰竭",
     "头晕、高钾血症、肾功能影响",
     "妊娠期、双侧肾动脉狭窄",
     "成人：80-160mg/日",
     "ARB类药物，咳嗽副作用少"),

    ("美托洛尔", "降压药",
     "高血压、心绞痛、心力衰竭、心律失常",
     "乏力、心动过缓、低血压、肢端发冷",
     "严重心动过缓、哮喘、II度以上房室传导阻滞",
     "成人：25-100mg/次，每日1-2次",
     "β受体阻滞剂，突然停药需缓慢减量"),

    ("氢氯噻嗪", "降压药",
     "高血压、水肿性疾病",
     "低钾血症、高尿酸血症、高血糖、血脂异常",
     "无尿、严重肾功能不全、磺胺类过敏",
     "成人：12.5-25mg/日",
     "噻嗪类利尿剂"),

    ("依那普利", "降压药",
     "高血压、心力衰竭、无症状性左心室功能不全",
     "干咳、头晕、低血压、高钾血症",
     "妊娠期、血管神经性水肿史",
     "成人：5-20mg/日，分1-2次",
     "ACEI类长效制剂"),

    ("氯沙坦", "降压药",
     "高血压、2型糖尿病肾病患者",
     "头晕、低血压、高钾血症",
     "妊娠期、严重肝损害",
     "成人：50-100mg/日",
     "ARB类药物"),

    # ── 降糖药 ──
    ("二甲双胍", "降糖药",
     "2型糖尿病，特别是肥胖患者",
     "胃肠道反应（恶心、腹泻）、乳酸酸中毒（罕见）",
     "严重肾功能不全、肝功能不全、心衰、酮症酸中毒",
     "成人：500-2000mg/日，分次服用",
     "2型糖尿病一线用药"),

    ("格列本脲", "降糖药",
     "2型糖尿病",
     "低血糖、体重增加、胃肠道不适",
     "1型糖尿病、酮症酸中毒、严重肝肾功能不全",
     "成人：2.5-15mg/日",
     "磺脲类促胰岛素分泌剂"),

    ("阿卡波糖", "降糖药",
     "2型糖尿病（降低餐后血糖）",
     "腹胀、排气增多、腹泻",
     "消化吸收障碍、严重肾功能不全",
     "成人：50-100mg/次，每日3次",
     "α-葡萄糖苷酶抑制剂"),

    # ── 降脂药 ──
    ("阿托伐他汀", "降脂药",
     "高胆固醇血症、冠心病、动脉粥样硬化",
     "肌肉疼痛、肝功能异常",
     "活动性肝病、妊娠期、哺乳期",
     "成人：10-80mg/日",
     "他汀类，睡前服用效果更佳"),

    ("辛伐他汀", "降脂药",
     "高胆固醇血症、冠心病高危人群",
     "肌肉酸痛、肝功能异常、头痛",
     "活动性肝病、妊娠期",
     "成人：10-40mg/日，晚间服用",
     "他汀类药物"),

    ("非诺贝特", "降脂药",
     "高甘油三酯血症、混合型高脂血症",
     "胃肠道不适、肝功能异常、胆结石",
     "严重肝肾功能不全、胆囊疾病",
     "成人：200mg/日",
     "贝特类药物"),

    # ── 消化系统 ──
    ("奥美拉唑", "消化系统药",
     "胃食管反流病、消化性溃疡、幽门螺杆菌根除",
     "头痛、腹泻、恶心、长期使用增加骨折风险",
     "对PPI过敏者",
     "成人：20mg/日，早餐前服用",
     "质子泵抑制剂"),

    ("多潘立酮", "消化系统药",
     "功能性消化不良、胃轻瘫、恶心呕吐",
     "口干、头痛、腹泻，偶见锥体外系反应",
     "胃肠道出血、穿孔、机械性梗阻、催乳素瘤",
     "成人：10mg/次，每日3次，餐前15-30分钟",
     "促胃动力药"),

    ("铝碳酸镁", "消化系统药",
     "胃酸过多、胃灼热、反酸性消化不良",
     "少数有软便或腹泻",
     "严重肾功能不全",
     "成人：500-1000mg/次，每日3-4次",
     "抗酸药，饭后1-2小时服用"),

    ("蒙脱石散", "消化系统药",
     "成人及儿童急慢性腹泻",
     "少数有便秘",
     "对本药过敏者",
     "成人：3g/次，每日3次",
     "肠道黏膜保护剂"),

    # ── 抗过敏 ──
    ("氯雷他定", "抗过敏药",
     "过敏性鼻炎、荨麻疹、皮肤过敏",
     "乏力、头痛、口干（发生率低）",
     "严重肝肾功能不全者调整剂量",
     "成人：10mg/日",
     "第二代抗组胺药，无嗜睡"),

    ("西替利嗪", "抗过敏药",
     "过敏性鼻炎、荨麻疹、过敏性结膜炎",
     "少数有嗜睡、口干、头痛",
     "严重肾功能不全者调整剂量",
     "成人：10mg/日",
     "第二代抗组胺药"),

    # ── 呼吸系统 ──
    ("氨溴索", "呼吸系统药",
     "急慢性呼吸道疾病伴痰液黏稠、排痰困难",
     "胃肠道反应、皮疹",
     "对本品过敏者",
     "成人：30-60mg/次，每日3次",
     "祛痰药"),

    ("右美沙芬", "呼吸系统药",
     "干咳、刺激性咳嗽",
     "头晕、嗜睡、恶心",
     "有精神病史、MAOI药物同用、妊娠早期",
     "成人：15-30mg/次，每日3-4次",
     "中枢性镇咳药，无成瘾性"),

    # ── 神经系统 ──
    ("地西泮", "神经系统药",
     "焦虑症、失眠、肌肉痉挛、癫痫持续状态",
     "嗜睡、乏力、共济失调、依赖成瘾",
     "重症肌无力、严重呼吸功能不全、青光眼",
     "成人：2.5-10mg/次，每日2-4次",
     "苯二氮卓类，长期使用可产生依赖性"),

    ("卡马西平", "神经系统药",
     "三叉神经痛、癫痫（部分性发作）、躁狂症",
     "头晕、共济失调、皮疹、肝功能异常、低钠血症",
     "房室传导阻滞、严重肝病、骨髓抑制",
     "成人：200-1200mg/日，分次服用",
     "需监测血药浓度"),

    # ── 抗凝药 ──
    ("华法林", "抗凝药",
     "预防血栓栓塞性疾病、人工心脏瓣膜术后",
     "出血（鼻出血、牙龈出血、皮肤瘀斑）",
     "出血倾向、严重高血压、妊娠期、近期手术",
     "根据INR调整剂量",
     "需定期监测凝血功能（INR维持在2-3）"),

    ("氯吡格雷", "抗凝药",
     "预防动脉粥样硬化血栓事件（心梗、卒中、外周动脉疾病）",
     "出血、胃肠道不适、皮疹",
     "活动性出血、严重肝病",
     "成人：75mg/日",
     "ADP受体拮抗剂"),

    # ── 其他 ──
    ("维生素C", "维生素类",
     "坏血病的预防和治疗，增强免疫力，促进铁吸收",
     "大剂量可致腹泻、尿路结石",
     "高草酸盐尿症、肾结石",
     "成人：100-200mg/日",
     "水溶性维生素"),

    ("维生素D", "维生素类",
     "维生素D缺乏、骨质疏松症",
     "大剂量可致高钙血症",
     "高钙血症、高钙尿症",
     "成人：400-800IU/日",
     "脂溶性维生素"),

    ("阿司匹林", "解热镇痛药",
     "用于缓解轻度至中度疼痛如头痛、牙痛、肌肉痛；也用于退热和抗血小板聚集",
     "胃肠道刺激、恶心、溃疡、出血时间延长、耳鸣",
     "活动性胃溃疡、出血体质、过敏者、妊娠晚期",
     "解热镇痛：300-600mg/次；抗血小板：75-100mg/日",
     "不要与布洛芬等NSAIDs同时服用，会增加胃肠道出血风险"),
]

INTERACTIONS_SEED = [
    ("阿司匹林", "布洛芬",
     "major",
     "两者均为NSAIDs，联用显著增加胃肠道出血和溃疡风险",
     "避免联合使用；如需抗血小板治疗，可考虑氯吡格雷替代阿司匹林"),
    ("阿司匹林", "华法林",
     "major",
     "阿司匹林抑制血小板功能，华法林抑制凝血因子，联用严重增加出血风险",
     "如需联用需密切监测INR，并评估胃肠道保护需求"),
    ("华法林", "阿莫西林",
     "moderate",
     "抗生素可能影响肠道菌群进而影响维生素K合成，增强华法林抗凝作用",
     "联用期间应增加INR监测频率"),
    ("二甲双胍", "碘造影剂",
     "major",
     "碘造影剂可能诱发急性肾损伤，二甲双胍在肾损伤时增加乳酸酸中毒风险",
     "检查前停用二甲双胍48小时，检查后确认肾功能正常再恢复用药"),
    ("卡托普利", "氢氯噻嗪",
     "moderate",
     "联用增强降压效果，但增加高钾血症风险（尤其是肾功能不全者）",
     "定期监测血钾和肾功能"),
    ("奥美拉唑", "氯吡格雷",
     "moderate",
     "奥美拉唑通过CYP2C19抑制氯吡格雷活化，可能降低氯吡格雷疗效",
     "可考虑用泮托拉唑或雷贝拉唑替代奥美拉唑"),
    ("左氧氟沙星", "布洛芬",
     "moderate",
     "NSAIDs增强喹诺酮类对GABA受体的抑制作用，增加中枢神经系统不良反应风险",
     "有癫痫病史者应避免联用"),
    ("地西泮", "酒精",
     "major",
     "酒精增强苯二氮卓类的中枢抑制作用，加重嗜睡、呼吸抑制",
     "服药期间严格禁酒"),
    ("硝苯地平", "克拉霉素",
     "moderate",
     "克拉霉素抑制CYP3A4代谢酶，使硝苯地平血药浓度升高，增加低血压风险",
     "联用期间监测血压，必要时减少硝苯地平剂量"),
    ("华法林", "阿奇霉素",
     "moderate",
     "阿奇霉素可能增强华法林抗凝作用",
     "联用期间增加INR监测频率"),
    ("甲硝唑", "华法林",
     "moderate",
     "甲硝唑抑制华法林代谢，显著增强抗凝作用",
     "联用期间密切监测INR，预防出血"),
    ("卡马西平", "华法林",
     "moderate",
     "卡马西平诱导肝药酶，加速华法林代谢，降低抗凝效果",
     "联用期间监测INR，可能需要增加华法林剂量"),
    ("缬沙坦", "螺内酯",
     "moderate",
     "两者均升高血钾，联用增加高钾血症风险",
     "定期监测血钾，避免过量补钾"),
    ("氯雷他定", "克拉霉素",
     "minor",
     "克拉霉素可能升高氯雷他定血药浓度",
     "通常无需调整剂量，注意观察不良反应"),
    ("阿司匹林", "氨氯地平",
     "minor",
     "氨氯地平可能增加阿司匹林的抗血小板作用",
     "常规监测，通常无需调整剂量"),
]


# ── 数据库初始化 ─────────────────────────────────────

async def _ensure_db():
    """确保数据库已就绪，首次使用自动创建并填充种子数据"""
    if os.path.exists(DB_PATH):
        # 检查是否已有数据
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM drugs")
            count = (await cursor.fetchone())[0]
            if count > 0:
                return

    logger.info("首次使用药品数据库 — 创建并初始化 %s", DB_PATH)
    await _init_db()


async def _init_db():
    """创建表并插入种子数据"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS drugs (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                category TEXT,
                indications TEXT,
                side_effects TEXT,
                contraindications TEXT,
                dosage TEXT,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS drug_interactions (
                id INTEGER PRIMARY KEY,
                drug_a TEXT NOT NULL,
                drug_b TEXT NOT NULL,
                severity TEXT,
                description TEXT,
                recommendation TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_drugs_name ON drugs(name);
            CREATE INDEX IF NOT EXISTS idx_interactions_drugs ON drug_interactions(drug_a, drug_b);
        """)
        # 插入药品数据
        await db.executemany(
            "INSERT OR IGNORE INTO drugs (name, category, indications, side_effects, contraindications, dosage, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            DRUGS_SEED,
        )
        # 插入相互作用数据
        await db.executemany(
            "INSERT OR IGNORE INTO drug_interactions (drug_a, drug_b, severity, description, recommendation) "
            "VALUES (?, ?, ?, ?, ?)",
            INTERACTIONS_SEED,
        )
        await db.commit()

    logger.info("药品数据库初始化完成: %d 种药品, %d 条相互作用",
                 len(DRUGS_SEED), len(INTERACTIONS_SEED))


# ── 工具处理器 ────────────────────────────────────────

async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 工具调度入口"""
    await _ensure_db()

    if name == "get_drug_info":
        return await _get_drug_info(arguments.get("drug_name", ""))
    elif name == "check_drug_interaction":
        return await _check_drug_interaction(
            arguments.get("drug_a", ""),
            arguments.get("drug_b", ""),
        )
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _get_drug_info(drug_name: str) -> Dict[str, Any]:
    """查询药品详细信息"""
    if not drug_name:
        return {"found": False, "message": "请提供药品名称", "suggestions": []}

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM drugs WHERE name = ?", (drug_name,))
        row = await cursor.fetchone()

        if row is None:
            # 模糊搜索
            cursor = await db.execute(
                "SELECT name FROM drugs WHERE name LIKE ? LIMIT 6",
                (f"%{drug_name}%",),
            )
            suggestions = [r[0] for r in await cursor.fetchall()]
            msg = f"未找到药品「{drug_name}」"
            if suggestions:
                msg += f"，您是否要找：{'、'.join(suggestions)}？"
            return {"found": False, "message": msg, "suggestions": suggestions}

        return {
            "found": True,
            "drug": {
                "name": row["name"],
                "category": row["category"],
                "indications": row["indications"],
                "side_effects": row["side_effects"],
                "contraindications": row["contraindications"],
                "dosage": row["dosage"],
                "notes": row["notes"],
            },
        }


async def _check_drug_interaction(drug_a: str, drug_b: str) -> Dict[str, Any]:
    """检查两种药品之间的相互作用"""
    if not drug_a or not drug_b:
        return {"found": False, "message": "请提供两种药品名称"}

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 验证药品是否存在
        cursor = await db.execute(
            "SELECT name FROM drugs WHERE name IN (?, ?)", (drug_a, drug_b)
        )
        existing = {r[0] for r in await cursor.fetchall()}

        not_found = []
        if drug_a not in existing:
            not_found.append(drug_a)
        if drug_b not in existing:
            not_found.append(drug_b)

        if not_found:
            return {"found": False, "message": f"数据库中未找到：{'、'.join(not_found)}"}

        # 查询相互作用（双向）
        cursor = await db.execute(
            """SELECT severity, description, recommendation FROM drug_interactions
               WHERE (drug_a = ? AND drug_b = ?) OR (drug_a = ? AND drug_b = ?)""",
            (drug_a, drug_b, drug_b, drug_a),
        )
        row = await cursor.fetchone()

        if row is None:
            return {
                "found": True,
                "has_interaction": False,
                "message": f"「{drug_a}」和「{drug_b}」之间未发现已知相互作用，但仍建议在医师指导下使用。",
                "severity": None,
                "description": None,
                "recommendation": None,
            }

        severity_labels = {"major": "严重", "moderate": "中等", "minor": "轻微"}
        return {
            "found": True,
            "has_interaction": True,
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": row["severity"],
            "severity_label": severity_labels.get(row["severity"], row["severity"]),
            "description": row["description"],
            "recommendation": row["recommendation"],
        }
