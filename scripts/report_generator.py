from openai import OpenAI
import os

api_key = os.environ.get("DASHSCOPE_API_KEY", "your-key")
client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def generate_warning_report(paths, predictions, anomalies, df):
    path_summary = "\n".join([f"路径: {' -> '.join(p['path'])} (攻击概率: {p['risk_prob']:.2f})" for p in paths])
    anomaly_summary = "\n".join([f"日期: {df.index[-30+i].date()}, 攻击次数: {df['攻击次数'].iloc[-30+i]}" for i, a in enumerate(anomalies) if a == -1])
    prompt = f"""
    根据历史威胁情报和当前趋势，预测未来7天的攻击次数为{predictions}。
    检测到的异常：
    {anomaly_summary}
    预测的高风险路径：
    {path_summary}
    请生成一份安全威胁报告，并建议应对措施。
    """
    try:
        res = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ 报告生成失败：{e}"
