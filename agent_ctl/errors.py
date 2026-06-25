# agent_ctl/errors.py
class GatewayError(Exception):
    """网关层基异常。"""


class RetriableError(GatewayError):
    """可重试(429/5xx/overloaded/timeout)。"""


class DeadlineExceeded(RetriableError):
    """单次调用墙钟总预算耗尽。是 RetriableError 子类(仍触发回退/停止),但**不计入熔断**
    ——deadline 耗尽是调用方时间预算问题,非 provider 健康问题,不应短路该 provider。"""


class TerminalError(GatewayError):
    """终态(4xx 鉴权/参数),不重试,直接透传。"""


class AllTargetsFailed(GatewayError):
    """路由链全部目标耗尽。"""


class BudgetExceeded(GatewayError):
    """成本预算闸:consumer/全局已达上限,调用前短路(不产生真实开销)。"""
