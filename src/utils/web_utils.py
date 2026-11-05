import json
import os
import re
import urllib.error
import urllib.request
import asyncio
import logging
from typing import List, Dict, Any
import ollama
from tavily import TavilyClient

class WebSearcher:
    def __init__(self, web_cfg: Dict[str, Any], ollama_cfg: Dict[str, Any], gen_cfg: Dict[str, Any]):
        self.web_cfg = web_cfg
        self.ollama_cfg = ollama_cfg
        self.gen_cfg = gen_cfg
        
        # 搜索服务可选 tavily 或 deepseek，密钥来自配置或本地覆盖文件
        self.provider = str(self.web_cfg.get("PROVIDER", "tavily")).lower()
        self.deepseek_base = str(self.web_cfg.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        if self.provider == "deepseek":
            self.api_key = str(self.web_cfg.get("DEEPSEEK_API_KEY", "") or "")
            self.tavily_client = None
        else:
            self.api_key = str(self.web_cfg.get("TAVILY_API_KEY", "") or "")
            self.tavily_client = TavilyClient(api_key=self.api_key) if self.api_key else None
        self.enabled = bool(self.api_key)
            
        self.extraction_model = self.web_cfg["EXTRACTION_MODEL"]
        self.explanation_model = self.web_cfg["EXPLANATION_MODEL"]
        
        # 合并推理参数
        self.ollama_options = self.ollama_cfg["options"].copy()
        self.ollama_options["temperature"] = gen_cfg["temperature"]
        self.ollama_options["num_predict"] = gen_cfg["max_new_tokens"]
        
        self.keep_alive = self.ollama_cfg["keep_alive"]
        self.use_think = self.ollama_cfg["think"]

    def _search_deepseek(self, query: str) -> str:
        """调用 DeepSeek 搜索接口，返回拼接后的摘要文本"""
        body = json.dumps({
            "query": query[:120],
            "count": max(1, min(50, int(self.web_cfg.get("MAX_RESULTS", 3)))),
            "need_content": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.deepseek_base.rstrip("/") + "/v1/tools/search",
            data=body,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        parts = []
        for item in data.get("results", []):
            text = str(item.get("summary") or item.get("snippet") or "").strip()
            if text:
                parts.append("- " + text)
        return "\n".join(parts)

    def _search_tavily(self, query: str) -> str:
        """调用 Tavily 搜索接口，返回拼接后的摘要文本"""
        response = self.tavily_client.search(
            query=query,
            search_depth=self.web_cfg.get("SEARCH_DEPTH", "basic"),
            max_results=int(self.web_cfg.get("MAX_RESULTS", 3)),
            include_answer=False,
        )
        parts = ["- " + res.get("content", "").strip()
                 for res in response.get("results", []) if res.get("content")]
        return "\n".join(parts)

    def _search(self, query: str) -> str:
        """按配置选择搜索服务，未配置密钥时返回空串"""
        if not self.enabled:
            logging.warning("未配置搜索密钥，跳过联网检索")
            return ""
        if self.provider == "deepseek":
            return self._search_deepseek(query)
        return self._search_tavily(query)

    def extract_unknown_words(self, comment: str) -> List[str]:
        """使用 LLM 提取疑似词"""
        prompt = self.web_cfg["prompts"]["extract_unknown_words"].format(comment=comment)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = ollama.chat(
                model=self.extraction_model,
                messages=messages,
                stream=False,
                options=self.ollama_options,
                keep_alive=self.keep_alive,
                think=False, # 提取词通常不需要思维链
                format=None # 暂时取消 format=json，手动清洗
            )
            content = response.message.content
            # 手动提取 JSON 列表
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                words = json.loads(match.group(0))
                return words if isinstance(words, list) else []
            return []
        except Exception as e:
            logging.error(f"提取疑似词失败: {e}")
            return []

    def search_term(self, term: str) -> str:
        """执行网络搜索"""
        query = f"{term} 是什么梗 网络用语意思"
        try:
            return self._search(query)
        except Exception as e:
            logging.error(f"搜索词汇 '{term}' 失败: {e}")
            return ""

    def search_sentence(self, sentence: str) -> str:
        """执行句子级别的背景搜索"""
        # 句子搜索通常需要更宽泛的上下文，不需要硬编码“是什么梗”
        query = f"{sentence} 背景 意思"
        return self._search(query)

    def explain_term(self, term: str, search_context: str) -> Dict[str, str]:
        """使用 LLM 总结词义"""
        if not search_context:
            return {"term": term, "definition": "未找到搜索结果", "sentiment": "未知"}

        prompt = self.web_cfg["prompts"]["explain_unknown_word"].format(term=term, search_context=search_context)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = ollama.chat(
                model=self.explanation_model,
                messages=messages,
                stream=False,
                options=self.ollama_options,
                keep_alive=self.keep_alive,
                think=self.use_think,
                format="json"
            )
            content = response.message.content
            return json.loads(content)
        except Exception as e:
            logging.error(f"总结词汇 '{term}' 失败: {e}")
            return {"term": term, "definition": "系统错误", "sentiment": "未知"}

    def explain_sentence(self, sentence: str, search_context: str) -> Dict[str, Any]:
        """使用 LLM 总结句子的背景"""
        if not search_context:
            raise ValueError(f"搜索句子背景未找到结果")

        prompt = self.web_cfg["prompts"]["explain_sentence_context"].format(sentence=sentence, search_context=search_context)
        messages = [{"role": "user", "content": prompt}]
        
        response = ollama.chat(
            model=self.explanation_model,
            messages=messages,
            stream=False,
            options=self.ollama_options,
            keep_alive=self.keep_alive,
            think=self.use_think,
            format="json"
        )
        content = response.message.content
        result = json.loads(content)
        # 严格要求字段存在
        return {
            "background": result["background"],
            "entities": result["entities"],
            "sentiment": result["sentiment"]
        }

    def run_web_search_flow(self, comment: str) -> str:
        """运行完整的网页搜索工作流"""
        # 1. 提取
        terms = self.extract_unknown_words(comment)
        if not terms:
            return ""
            
        # 2. 搜索与总结 (这里可以考虑并发，但为了稳定先串行)
        summaries = []
        for term in terms:
            context = self.search_term(term)
            if context:
                summary = self.explain_term(term, context)
                summaries.append(f"词汇: {summary.get('term')}\n解释: {summary.get('definition')}\n倾向: {summary.get('sentiment')}")
        
        if not summaries:
            return ""
            
        return "\n\n".join(summaries)

    def run_sentence_search_flow(self, sentence: str) -> str:
        """运行句子的搜索工作流"""
        context = self.search_sentence(sentence)
        if not context:
            return ""
            
        summary = self.explain_sentence(sentence, context)
        entities_str = ", ".join(summary.get("entities", []))
        return f"背景: {summary.get('background')}\n涉及实体: {entities_str}\n整体倾向: {summary.get('sentiment')}"
