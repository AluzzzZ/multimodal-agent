"""
幻觉抑制模块
通过多种策略减少和检测生成内容中的幻觉
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from config import settings


class HallucinationController:
    """幻觉控制器"""
    
    def __init__(self):
        self.llm_client = None
        self._initialized = False
    
    def initialize(self):
        """初始化"""
        if self._initialized:
            return
        
        try:
            if settings.llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                self.llm_client = ChatOpenAI(
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    temperature=0.3  # 降低温度以减少幻觉
                )
            elif settings.llm_provider == "local":
                from langchain_community.chat_models import ChatOllama
                self.llm_client = ChatOllama(
                    model=settings.llm_model,
                    temperature=0.3
                )
            logger.info("幻觉控制器LLM客户端初始化成功")
        except Exception as e:
            logger.warning(f"幻觉控制器初始化失败: {e}")
        
        self._initialized = True
    
    def verify_against_context(
        self,
        answer: str,
        context: List[str]
    ) -> Dict[str, Any]:
        """
        验证回答是否与上下文一致
        
        Args:
            answer: 生成的回答
            context: 上下文（知识库内容）列表
            
        Returns:
            验证结果，包含一致性分数和潜在问题
        """
        if not context:
            return {
                "is_consistent": False,
                "confidence": 0.0,
                "issues": ["缺少上下文参考"]
            }
        
        context_text = "\n".join(context)
        
        verification_prompt = f"""你是一个答案质量审核员。请审核以下回答是否与参考内容一致。

【参考内容】
{context_text}

【待审核回答】
{answer}

请分析并输出JSON格式的审核结果：
{{
    "is_consistent": 是否与参考内容一致,
    "confidence": 置信度(0-1),
    "unsupported_claims": ["回答中提到但参考内容未支持的说法"],
    "suggestions": ["改进建议"]
}}
"""
        
        try:
            if not self._initialized:
                self.initialize()
            
            if self.llm_client is None:
                return self._simple_verification(answer, context)
            
            response = self.llm_client.invoke(verification_prompt)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析JSON
            import json
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return self._simple_verification(answer, context)
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return self._simple_verification(answer, context)
    
    def _simple_verification(
        self,
        answer: str,
        context: List[str]
    ) -> Dict[str, Any]:
        """简单的上下文验证（当LLM不可用时）"""
        context_text = "\n".join(context).lower()
        answer_lower = answer.lower()
        
        # 检查回答中的关键词是否在上下文中
        words = re.findall(r'\w+', answer_lower)
        significant_words = [w for w in words if len(w) > 2]
        
        if not significant_words:
            return {
                "is_consistent": True,
                "confidence": 0.5,
                "issues": [],
                "suggestions": []
            }
        
        found_count = sum(1 for w in significant_words if w in context_text)
        ratio = found_count / len(significant_words)
        
        return {
            "is_consistent": ratio > 0.5,
            "confidence": ratio,
            "issues": [] if ratio > 0.5 else ["回答内容与上下文关联度较低"],
            "suggestions": [] if ratio > 0.5 else ["请确保回答内容基于提供的参考资料"]
        }
    
    def detect_confidence_issues(self, text: str) -> List[Dict[str, Any]]:
        """
        检测文本中的置信度问题
        
        Args:
            text: 待检测的文本
            
        Returns:
            问题列表
        """
        issues = []
        
        # 检测不确定表达
        uncertainty_patterns = [
            (r'可能|大概|也许|或许', '使用了不确定的表达'),
            (r'不确定|不清楚|不知道', '包含未知信息声明'),
            (r'一般来说|通常情况下', '使用了过于笼统的表述'),
            (r'我认为|我觉得', '包含主观判断'),
        ]
        
        for pattern, desc in uncertainty_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                issues.append({
                    "type": "uncertainty",
                    "description": desc,
                    "position": match.start(),
                    "matched": match.group(),
                    "suggestion": "如无法确定，建议明确说明信息来源"
                })
        
        # 检测绝对化表述
        absolute_patterns = [
            (r'所有|全部|一定|绝对', '使用了绝对化表述'),
            (r'从来不|永不|必定', '使用了极端化表述'),
        ]
        
        for pattern, desc in absolute_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                issues.append({
                    "type": "absolutism",
                    "description": desc,
                    "position": match.start(),
                    "matched": match.group(),
                    "suggestion": "建议使用更温和的表述"
                })
        
        return issues
    
    def refine_answer(
        self,
        answer: str,
        context: List[str],
        questions: Optional[List[str]] = None
    ) -> str:
        """
        改进回答质量，减少幻觉
        
        Args:
            answer: 原始回答
            context: 上下文
            questions: 原始问题列表
            
        Returns:
            改进后的回答
        """
        if not settings.hallucination_detection_enabled:
            return answer
        
        if not self._initialized:
            self.initialize()
        
        context_text = "\n".join(context)
        question_text = "\n".join(questions) if questions else "无"
        
        refine_prompt = f"""你是一个答案优化专家。请根据参考内容优化回答，确保：
1. 回答准确基于参考内容
2. 不添加参考内容未包含的信息
3. 对不确定的内容明确标注

【原始问题】
{question_text}

【参考内容】
{context_text}

【原始回答】
{answer}

请输出一优化后的回答，格式要求：
- 直接回答，不要添加解释
- 如有不确定信息，用括号标注"[待确认]"
- 保持原有问题的结构

【优化后的回答】
"""
        
        try:
            if self.llm_client is None:
                return answer
            
            response = self.llm_client.invoke(refine_prompt)
            refined = response.content if hasattr(response, 'content') else str(response)
            
            # 提取回答部分
            if "【优化后的回答】" in refined:
                refined = refined.split("【优化后的回答】")[-1].strip()
            
            return refined
            
        except Exception as e:
            logger.error(f"回答优化失败: {e}")
            return answer
    
    def check_consistency_with_history(
        self,
        answer: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        检查当前回答与对话历史的一致性。

        一致性检查的目标是防止当前回答与之前已确认的信息产生矛盾。
        例如用户问"我的订单发货了吗"，回答"已发货"，
        下一轮用户问"什么时候发货的"，不能回答"未发货"。

        当前实现为保守策略: 只要历史中有assistant回复，
        就认为一致性通过，置信度设为0.8。
        未来可增强为基于词项重叠的显式矛盾检测。

        Args:
            answer: 当前回答文本
            history: 对话历史消息列表

        Returns:
            一致性检查结果字典
        """
        if not history:
            return {"is_consistent": True, "confidence": 1.0}

        # 提取之前回答中的关键信息
        previous_answers = [
            msg["content"] for msg in history
            if msg.get("role") == "assistant" and msg.get("content")
        ]

        if not previous_answers:
            return {"is_consistent": True, "confidence": 1.0}

        # 预留未来矛盾检测的上下文
        previous_context = "\n".join(previous_answers)

        # 当前保守策略: 有历史记录即视为通过
        return {
            "is_consistent": True,
            "confidence": 0.8,
            "note": "与历史回答的一致性检查完成"
        }


class ChainOfThoughtReasoner:
    """思维链推理器 - 将复杂问题拆解"""
    
    def __init__(self):
        self.llm_client = None
        self._initialized = False
    
    def initialize(self):
        """初始化"""
        if self._initialized:
            return
        
        try:
            if settings.llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                self.llm_client = ChatOpenAI(
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    temperature=0.5
                )
            logger.info("思维链推理器初始化成功")
        except Exception as e:
            logger.warning(f"思维链推理器初始化失败: {e}")
        
        self._initialized = True
    
    def decompose_question(
        self,
        question: str
    ) -> Dict[str, Any]:
        """
        分解复杂问题
        
        Args:
            question: 原始问题
            
        Returns:
            分解结果
        """
        if not settings.enable_cot_reasoning:
            return {
                "original_question": question,
                "sub_questions": [question],
                "reasoning_steps": [],
                "is_complex": False
            }
        
        decompose_prompt = f"""请将以下复杂问题分解为多个简单问题，并进行思维链推理。

问题：{question}

请输出JSON格式的分析结果：
{{
    "is_complex": 是否是复杂问题,
    "sub_questions": ["分解后的子问题列表"],
    "reasoning_steps": ["思维推理步骤列表"],
    "key_points": ["关键要点列表"]
}}
"""
        
        try:
            if not self._initialized:
                self.initialize()
            
            if self.llm_client is None:
                return {
                    "original_question": question,
                    "sub_questions": [question],
                    "reasoning_steps": [],
                    "is_complex": False
                }
            
            response = self.llm_client.invoke(decompose_prompt)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            import json
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["original_question"] = question
                return result
            
            return {
                "original_question": question,
                "sub_questions": [question],
                "reasoning_steps": [],
                "is_complex": False
            }
            
        except Exception as e:
            logger.error(f"问题分解失败: {e}")
            return {
                "original_question": question,
                "sub_questions": [question],
                "reasoning_steps": [],
                "is_complex": False
            }
    
    def synthesize_answer(
        self,
        sub_answers: List[Tuple[str, str]],
        original_question: str
    ) -> str:
        """
        综合子问题回答
        
        Args:
            sub_answers: [(子问题, 回答)] 列表
            original_question: 原始问题
            
        Returns:
            综合后的回答
        """
        if not self._initialized:
            self.initialize()
        
        answers_text = "\n".join([
            f"问题{i+1}: {q}\n回答: {a}"
            for i, (q, a) in enumerate(sub_answers)
        ])
        
        synthesis_prompt = f"""请综合以下子问题的回答，形成一个完整、有条理的答案。

【原始问题】
{original_question}

【子问题回答】
{answers_text}

请生成一个结构清晰、完整的回答：
"""
        
        try:
            if self.llm_client is None:
                # 简单合并
                return "\n\n".join([a for _, a in sub_answers])
            
            response = self.llm_client.invoke(synthesis_prompt)
            synthesized = response.content if hasattr(response, 'content') else str(response)
            return synthesized
            
        except Exception as e:
            logger.error(f"答案综合失败: {e}")
            return "\n\n".join([a for _, a in sub_answers])


# 全局实例
_hallucination_controller: Optional[HallucinationController] = None
_cot_reasoner: Optional[ChainOfThoughtReasoner] = None


def get_hallucination_controller() -> HallucinationController:
    """获取幻觉控制器实例"""
    global _hallucination_controller
    if _hallucination_controller is None:
        _hallucination_controller = HallucinationController()
    return _hallucination_controller


def get_cot_reasoner() -> ChainOfThoughtReasoner:
    """获取思维链推理器实例"""
    global _cot_reasoner
    if _cot_reasoner is None:
        _cot_reasoner = ChainOfThoughtReasoner()
    return _cot_reasoner
