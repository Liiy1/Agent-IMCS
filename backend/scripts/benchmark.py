"""
问诊系统压测脚本
发送 200 次问诊请求，采集延迟数据，最后从 /stats 拉取聚合统计。
"""
import asyncio
import httpx
import time
import statistics
import sys
from typing import List, Dict

BASE_URL = "http://localhost:8000"
CONSULT_URL = f"{BASE_URL}/api/v1/consultation/consult"
STATS_URL = f"{BASE_URL}/api/v1/consultation/stats"

# 200 条多样化的症状描述
SYMPTOMS = [
    # ── 单症状 ──
    "头痛", "发烧", "咳嗽", "鼻塞", "喉咙痛",
    "肚子痛", "胸痛", "关节痛", "腰痛", "头晕",
    "乏力", "恶心", "呕吐", "腹泻", "便秘",
    "失眠", "心慌", "气短", "耳鸣", "眼睛干涩",

    # ── 双症状 ──
    "头痛三天，有点发烧", "咳嗽有痰，喉咙痛", "肚子痛还拉肚子",
    "发烧怕冷，全身酸痛", "头痛恶心，看东西模糊",
    "胸闷气短，心慌", "关节肿痛，早上僵硬",
    "腰酸背痛，腿发麻", "头晕耳鸣，睡眠不好",
    "胃痛反酸，吃完饭更难受",

    # ── 三症状 ──
    "头痛三天了，还有发烧、恶心和乏力",
    "咳嗽一周，咳黄痰，还有点发烧",
    "肚子痛拉肚子两天，还有点恶心发烧",
    "头晕目眩，耳鸣，走路不稳",
    "胸痛憋气，稍微活动就喘，心慌",
    "关节疼痛、肿胀、早上僵硬",
    "口干舌燥，喝很多水还是渴，尿多",
    "发烧39度，嗓子疼，全身酸痛",
    "头痛恶心想吐，怕光怕声音",
    "胃不舒服，腹胀，反酸烧心",

    # ── 慢性病史 ──
    "我有高血压，最近头痛头晕",
    "我是糖尿病，最近伤口不愈合",
    "我有高血脂，最近胸闷",
    "我是乙肝携带者，最近乏力",
    "我有胃溃疡病史，最近又胃痛了",
    "做过心脏支架，现在胸痛",
    "有哮喘史，最近喘得厉害",
    "甲状腺功能减退，最近很疲劳",
    "有肾结石史，最近腰疼",
    "类风湿关节炎，最近关节痛加重",

    # ── 详细描述 ──
    "头痛三天了，前额痛，胀痛，一阵一阵的，吃止痛药没用",
    "发烧两天，最高到39度，吃退烧药能退但反复",
    "咳嗽两周了，干咳没痰，晚上躺下更严重",
    "肚子痛在右下腹，按压更痛，还有点发烧",
    "胸口正中间痛，像压榨感，左手也麻",
    "头晕天旋地转，一翻身就晕，还恶心想吐",
    "腰疼半年了，久坐加重，弯腰困难",
    "眼睛痒痒发红，流眼泪打喷嚏",
    "嗓子疼三天，吞咽困难，有脓点",
    "拉肚子水样便，一天6次，肚子绞痛",

    # ── 儿科/老年 ──
    "宝宝发烧38度，不肯吃奶，精神不好",
    "老人咳嗽一个月，消瘦没胃口",
    "孩子身上起疹子，痒得睡不着",
    "孕妇恶心呕吐，什么都吃不下",
    "老年人记性越来越差，脾气也变了",
    "小孩肚子痛阵发性，哭闹不止",
    "老人走路喘，双下肢水肿",
    "孩子频繁眨眼清嗓子",
    "孕妇头痛眼花的，血压高",
    "老人摔倒后髋部痛，站不起来",

    # ── 各部位 ──
    "左边头痛像针扎", "右边肚子痛放射到后背",
    "手指关节对称性肿痛", "膝盖运动后弹响疼痛",
    "足跟痛早上下床第一步", "肩膀痛抬不起来",
    "肘关节外侧痛拧毛巾加重", "脚踝扭伤三个月还肿",
    "小腿抽筋夜间加重", "大腿根部走路痛",

    # ── 伴随症状 ──
    "发烧出皮疹", "头痛视力下降",
    "胸痛咳血", "腹痛便血",
    "腰痛尿频尿痛", "关节痛口腔溃疡",
    "头痛流脓鼻涕", "咳嗽痰中带血",
    "恶心想吐腹泻", "心慌手抖怕热",

    # ── 不同科室场景 ──
    "皮肤起红斑脱屑", "耳鸣听力下降",
    "牙痛脸肿了", "眼睛充血视力模糊",
    "鼻子堵流黄脓涕", "喉咙有异物感",
    "耳朵流脓听力下降", "口腔溃疡反复发作",
    "脱发严重头皮痒", "指甲变黄增厚",

    # ── 用户补充——第二轮症状 ──
    "头痛在太阳穴附近跳痛", "发烧时冷时热的",
    "咳嗽是刺激性干咳", "肚子痛在脐周",
    "胸痛深呼吸加重", "头晕是体位改变时明显",
    "腰疼放射到臀部", "关节痛是游走性的",
    "心慌一阵一阵的", "气短平躺加重坐起来好转",

    # ── 不同主诉长度 ──
    "头疼", "发烧了", "胃不舒服", "拉肚子", "睡不着",
    "嗓子不舒服", "全身没劲", "没胃口", "后背疼", "脖子疼",
    "脚崴了", "手麻", "脸肿", "尿频", "便血",
    "月经不调", "白带异常", "性生活出血", "腹痛痛经", "更年期潮热",

    # ── 紧急场景 ──
    "突然胸口剧痛大汗淋漓", "突发剧烈头痛呕吐",
    "呼吸困难嘴唇发紫", "大量便鲜血",
    "腹痛剧烈腹部硬如板",
    "高烧不退意识模糊",
    "外伤出血不止",
    "误食药物需要洗胃",
    "突然一侧肢体无力说话不清",
    "严重过敏全身皮疹呼吸困难",
]


async def send_request(client: httpx.AsyncClient, idx: int, message: str, results: List[Dict]):
    """发送一次问诊请求，记录延迟"""
    payload = {
        "session_id": f"bench-{idx}",
        "message": message,
        "user_token": "benchmark",
    }
    t0 = time.monotonic()
    try:
        resp = await client.post(CONSULT_URL, json=payload, timeout=120)
        elapsed = time.monotonic() - t0
        data = resp.json()
        results.append({
            "idx": idx,
            "status": resp.status_code,
            "ms": elapsed * 1000,
            "session_id": data.get("session_id", ""),
            "is_complete": data.get("is_complete", False),
            "urgency": data.get("urgency_level", ""),
            "response_len": len(data.get("response", "")),
        })
        if resp.status_code != 200:
            print(f"  [{idx}] ERROR: status={resp.status_code}, msg={message[:20]}")
        else:
            status = "✓" if data.get("is_complete") else "?"
            print(f"  [{idx}] {status} {elapsed:.1f}s urg={data.get('urgency_level','?'):8s} {message[:25]}")
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.append({"idx": idx, "status": 0, "ms": elapsed * 1000, "error": str(e)})
        print(f"  [{idx}] EXCEPTION: {e}")


async def main():
    print("=" * 60)
    print(f"问诊系统压测 — 共 {len(SYMPTOMS)} 次请求")
    print(f"目标: {BASE_URL}")
    print("=" * 60)

    # 健康检查
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"{BASE_URL}/health", timeout=5)
            print(f"健康检查: {r.json()}\n")
        except Exception as e:
            print(f"❌ 后端不可达: {e}")
            sys.exit(1)

    results: List[Dict] = []
    batch_size = 5  # 每批并发数（控制 DeepSeek 并发）
    delay_between = 0.5  # 批间延迟（秒）

    async with httpx.AsyncClient() as client:
        for i in range(0, len(SYMPTOMS), batch_size):
            batch = SYMPTOMS[i:i + batch_size]
            tasks = []
            for j, msg in enumerate(batch):
                idx = i + j
                tasks.append(send_request(client, idx, msg, results))

            await asyncio.gather(*tasks)
            if i + batch_size < len(SYMPTOMS):
                await asyncio.sleep(delay_between)

    # ── 统计 ──
    print("\n" + "=" * 60)
    print("📊 请求延迟统计")
    print("=" * 60)

    ok_results = [r for r in results if r["status"] == 200]
    fail_results = [r for r in results if r["status"] != 200]

    if ok_results:
        latencies = sorted([r["ms"] for r in ok_results])
        n = len(latencies)
        print(f"  成功: {n} / {len(results)}")
        print(f"  失败: {len(fail_results)}")
        print(f"  平均: {sum(latencies) / n:.0f} ms")
        print(f"  最短: {latencies[0]:.0f} ms")
        print(f"  最长: {latencies[-1]:.0f} ms")
        print(f"  P50:  {latencies[int(n * 0.50)]:.0f} ms")
        print(f"  P90:  {latencies[int(n * 0.90)]:.0f} ms")
        print(f"  P95:  {latencies[int(n * 0.95)]:.0f} ms")
        print(f"  P99:  {latencies[int(n * 0.99)]:.0f} ms")

        # 按 is_complete 分组
        complete = [r for r in ok_results if r["is_complete"]]
        incomplete = [r for r in ok_results if not r["is_complete"]]
        print(f"\n📊 工作流完成: {len(complete)} | 追问: {len(incomplete)}")

        # 按紧急度分组
        from collections import Counter
        urgency_counts = Counter(r["urgency"] for r in ok_results)
        if urgency_counts:
            print(f"\n📊 紧急度分布:")
            for level, count in urgency_counts.most_common():
                print(f"  {level}: {count}")

        # 响应长度分布
        response_lengths = [r["response_len"] for r in ok_results]
        if response_lengths:
            print(f"\n📊 报告长度:")
            print(f"  平均: {sum(response_lengths) / len(response_lengths):.0f} 字符")
            print(f"  最短: {min(response_lengths)} 字符")
            print(f"  最长: {max(response_lengths)} 字符")

    # ── 从 /stats 接口拉取服务端聚合数据 ──
    print("\n" + "=" * 60)
    print("📊 服务端 LatencyTracker 统计 (/stats)")
    print("=" * 60)
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(STATS_URL, timeout=5)
            stats = r.json()
            rs = stats.get("request_stats", {})
            if rs:
                print(f"  请求次数: {rs.get('count', 0)}")
                print(f"  平均: {rs.get('avg_ms', 0):.0f} ms")
                print(f"  P50:  {rs.get('p50_ms', 0):.0f} ms")
                print(f"  P90:  {rs.get('p90_ms', 0):.0f} ms")
                print(f"  P95:  {rs.get('p95_ms', 0):.0f} ms")
                print(f"  P99:  {rs.get('p99_ms', 0):.0f} ms")

            ls = stats.get("llm_stats", {})
            for method, s in ls.items():
                if s.get("count", 0):
                    print(f"  {method}: avg={s['avg_ms']:.0f}ms, P50={s['p50_ms']:.0f}ms, P95={s['p95_ms']:.0f}ms (n={s['count']})")
        except Exception as e:
            print(f"  (stats endpoint error: {e})")

    # ── 检查数据库多轮轮次 ──
    print("\n" + "=" * 60)
    print("📊 多轮轮次统计（从 messages 表）")
    print("=" * 60)
    # 注意：因本轮压测每 session 只发一条消息，多轮数据有限
    # 这里只统计有追问的 session
    incomplete_ids = set(r["session_id"] for r in incomplete if r.get("session_id"))
    if incomplete_ids:
        print(f"  产生追问的 session 数: {len(incomplete_ids)}")


if __name__ == "__main__":
    asyncio.run(main())
