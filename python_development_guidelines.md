# Python Development Guidelines 2

A comprehensive set of coding standards and best practices for Python development.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Object-Oriented Design](#object-oriented-design)
3. [Error Handling & Safety](#error-handling--safety)
4. [Immutability & Data Classes](#immutability--data-classes)
5. [Dependency Management](#dependency-management)
6. [Code Organization](#code-organization)
7. [Logging & Documentation](#logging--documentation)
8. [Collections & Resources](#collections--resources)
9. [Concurrency](#concurrency)
10. [Testing & Quality](#testing--quality)
11. [Reuse & Style](#reuse--style)
12. [Async & Reactive](#async--reactive)
13. [Type Hints](#type-hints)

---

## Core Principles

### 1. Law of Demeter

Only interact with immediate collaborators. Avoid chaining calls like `obj.get_a().get_b().get_c()`.

```python
# Avoid
user.get_account().get_settings().get_preferences().theme

# Prefer
user.get_theme_preference()
```

### 2. Single Responsibility Principle

Each class and function should have one clear purpose and one reason to change.

```python
# Avoid: Class doing too much
class UserManager:
    def create_user(self, data): ...
    def send_welcome_email(self, user): ...
    def generate_report(self, users): ...

# Prefer: Separate responsibilities
class UserRepository:
    def create(self, data) -> User: ...

class EmailService:
    def send_welcome_email(self, user: User) -> None: ...

class UserReportGenerator:
    def generate(self, users: list[User]) -> Report: ...
```

### 3. Early Returns (Guard Clauses)

Avoid deep nesting by using early returns to handle edge cases first.

```python
# Avoid
def process(data):
    if data:
        if data.is_valid:
            if data.has_items:
                # actual logic buried deep
                return result
    return None

# Prefer
def process(data):
    if not data:
        return None
    if not data.is_valid:
        raise ValidationError("Invalid data")
    if not data.has_items:
        return []
    
    # actual logic at the top level
    return result
```

### 4. Clean Code Principles

Write readable, concise, and easily understandable code. Prefer explicit over implicit.

```python
# Avoid
def calc(a, b, c):
    return a * (1 + b) ** c

# Prefer
def calculate_compound_amount(
    principal: Decimal,
    interest_rate: Decimal,
    periods: int,
) -> Decimal:
    return principal * (1 + interest_rate) ** periods
```

### 5. Meaningful Names

Variables, functions, and classes should clearly convey their purpose.

```python
# Avoid
d = get_data()
x = d[0]
temp = process(x)

# Prefer
user_records = fetch_user_records()
primary_user = user_records[0]
processed_user = enrich_user_profile(primary_user)
```

---

## Object-Oriented Design

### 6. Object-Oriented Principles

Follow encapsulation, abstraction, polymorphism, and prefer composition over inheritance.

```python
# Composition over inheritance
class EmailNotifier:
    def notify(self, message: str) -> None:
        # send email
        ...

class SMSNotifier:
    def notify(self, message: str) -> None:
        # send SMS
        ...

class NotificationService:
    def __init__(self, notifiers: list[Notifier]):
        self._notifiers = notifiers

    def notify_all(self, message: str) -> None:
        for notifier in self._notifiers:
            notifier.notify(message)
```

### 7. Design Patterns

Implement suitable patterns (Factory, Strategy, Repository, etc.) for simplicity and clarity.

```python
# Strategy Pattern
from typing import Protocol

class PricingStrategy(Protocol):
    def calculate_price(self, base_price: Decimal) -> Decimal: ...

class RegularPricing:
    def calculate_price(self, base_price: Decimal) -> Decimal:
        return base_price

class DiscountPricing:
    def __init__(self, discount_percent: Decimal):
        self._discount = discount_percent

    def calculate_price(self, base_price: Decimal) -> Decimal:
        return base_price * (1 - self._discount / 100)

class Order:
    def __init__(self, pricing_strategy: PricingStrategy):
        self._pricing = pricing_strategy

    def get_total(self, base_price: Decimal) -> Decimal:
        return self._pricing.calculate_price(base_price)
```

### 8. Protocols and Abstract Base Classes

Use `typing.Protocol` or `abc.ABC` for flexibility and programming to an interface.

```python
from typing import Protocol
from abc import ABC, abstractmethod

# Protocol (structural subtyping - duck typing with type hints)
class Repository(Protocol):
    def find_by_id(self, id: str) -> Entity | None: ...
    def save(self, entity: Entity) -> None: ...
    def delete(self, id: str) -> None: ...

# Abstract Base Class (nominal subtyping)
class BaseRepository(ABC):
    @abstractmethod
    def find_by_id(self, id: str) -> Entity | None:
        pass

    @abstractmethod
    def save(self, entity: Entity) -> None:
        pass
```

---

## Error Handling & Safety

### 9. Specific Exception Handling

Avoid catching generic exceptions. Handle specific exception types.

```python
# Avoid
try:
    process()
except Exception:
    pass

# Avoid
try:
    process()
except Exception as e:
    logger.error(e)

# Prefer
try:
    process()
except ValidationError as e:
    logger.warning("Validation failed: %s", e)
    raise
except ConnectionError as e:
    logger.error("Connection failed: %s", e)
    raise ServiceUnavailableError(f"Failed to connect: {e}") from e
except TimeoutError as e:
    logger.error("Operation timed out: %s", e)
    raise
```

### 10. Custom Exception Hierarchy

Define domain-specific exceptions for clear error handling.

```python
class AppError(Exception):
    """Base exception for application errors."""
    pass

class ValidationError(AppError):
    """Raised when input validation fails."""
    pass

class NotFoundError(AppError):
    """Raised when a requested resource is not found."""
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} with id '{resource_id}' not found")

class DuplicateError(AppError):
    """Raised when attempting to create a duplicate resource."""
    pass
```

### 11. None Safety

Use Optional types and guard against None values explicitly.

```python
from typing import Optional

def find_user(user_id: str) -> Optional[User]:
    """Find a user by ID, returns None if not found."""
    ...

# Always check before use
user = find_user(user_id)
if user is None:
    raise UserNotFoundError("User", user_id)

# Or use early return
def get_user_email(user_id: str) -> Optional[str]:
    user = find_user(user_id)
    if user is None:
        return None
    return user.email
```

---

## Immutability & Data Classes

### 12. Favor Immutability

Use frozen dataclasses or NamedTuples for thread safety and clarity.

```python
from dataclasses import dataclass, field
from decimal import Decimal
from typing import NamedTuple

# Frozen dataclass (immutable)
@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(amount=self.amount + other.amount, currency=self.currency)

# NamedTuple (also immutable)
class Coordinate(NamedTuple):
    latitude: float
    longitude: float

# Immutable with default factory
@dataclass(frozen=True)
class Order:
    id: str
    items: tuple[OrderItem, ...] = field(default_factory=tuple)
```

---

## Dependency Management

### 13. Dependency Injection

Pass dependencies explicitly rather than creating them internally.

```python
# Avoid: Hard-coded dependencies
class UserService:
    def __init__(self):
        self.repo = UserRepository()
        self.email_client = SMTPClient()

# Prefer: Injected dependencies
class UserService:
    def __init__(
        self,
        repo: UserRepository,
        email_service: EmailService,
    ):
        self._repo = repo
        self._email_service = email_service

    def create_user(self, data: CreateUserRequest) -> User:
        user = User.from_request(data)
        self._repo.save(user)
        self._email_service.send_welcome(user)
        return user

# Easy to test with mocks
def test_create_user():
    mock_repo = Mock(spec=UserRepository)
    mock_email = Mock(spec=EmailService)
    service = UserService(repo=mock_repo, email_service=mock_email)
    
    user = service.create_user(CreateUserRequest(name="Alice"))
    
    mock_repo.save.assert_called_once()
    mock_email.send_welcome.assert_called_once()
```

---

## Code Organization

### 14. Code Decomposition

Break lengthy or complex functions into smaller, focused functions.

```python
# Avoid: Long function doing many things
def process_order(order_data: dict) -> Order:
    # 100+ lines of validation, processing, saving, notifications...
    ...

# Prefer: Decomposed into focused functions
def process_order(order_data: dict) -> Order:
    validated_data = validate_order_data(order_data)
    order = create_order_from_data(validated_data)
    saved_order = save_order(order)
    send_order_confirmation(saved_order)
    return saved_order

def validate_order_data(data: dict) -> ValidatedOrderData:
    """Validate and sanitize order input data."""
    ...

def create_order_from_data(data: ValidatedOrderData) -> Order:
    """Create an Order entity from validated data."""
    ...

def save_order(order: Order) -> Order:
    """Persist order to the database."""
    ...

def send_order_confirmation(order: Order) -> None:
    """Send confirmation email to customer."""
    ...
```

### 15. Avoid Magic Numbers

Use constants with descriptive names.

```python
# Avoid
if retry_count > 3:
    raise MaxRetriesExceeded()

if response.status_code == 429:
    await asyncio.sleep(60)

# Prefer
MAX_RETRY_ATTEMPTS = 3
RATE_LIMIT_STATUS_CODE = 429
RATE_LIMIT_BACKOFF_SECONDS = 60

if retry_count > MAX_RETRY_ATTEMPTS:
    raise MaxRetriesExceeded()

if response.status_code == RATE_LIMIT_STATUS_CODE:
    await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)
```

### 16. Eliminate Duplicate Code

Extract common logic into reusable functions or classes.

```python
# Avoid: Duplicated validation logic
def create_user(email: str):
    if not email or "@" not in email:
        raise ValidationError("Invalid email")
    ...

def update_user_email(user_id: str, email: str):
    if not email or "@" not in email:
        raise ValidationError("Invalid email")
    ...

# Prefer: Extracted validation
def validate_email(email: str) -> str:
    """Validate and normalize an email address."""
    if not email or "@" not in email:
        raise ValidationError("Invalid email")
    return email.lower().strip()

def create_user(email: str):
    validated_email = validate_email(email)
    ...

def update_user_email(user_id: str, email: str):
    validated_email = validate_email(email)
    ...
```

---

## Logging & Documentation

### 17. Proper Logging

Use the `logging` module for events, warnings, and errors.

```python
import logging

logger = logging.getLogger(__name__)

class OrderProcessor:
    def process(self, order: Order) -> ProcessedOrder:
        logger.info("Processing order order_id=%s", order.id)
        
        try:
            result = self._do_processing(order)
            logger.info(
                "Order processed successfully order_id=%s total=%s",
                order.id,
                result.total,
            )
            return result
        except PaymentError as e:
            logger.warning(
                "Payment failed for order order_id=%s error=%s",
                order.id,
                e,
            )
            raise
        except Exception:
            logger.exception("Unexpected error processing order order_id=%s", order.id)
            raise
```

### 18. Comments

Code should be self-documenting. Use comments sparingly for complex logic only.

```python
# Avoid: Obvious comments
# Increment counter by 1
counter += 1

# Get the user from the database
user = user_repo.find_by_id(user_id)

# Prefer: Comments for non-obvious logic
# Using binary search here because the list is sorted and can contain
# millions of items; linear search would be too slow
index = bisect.bisect_left(sorted_items, target)

# The API returns dates in Unix timestamp format with milliseconds,
# but our domain expects seconds
timestamp_seconds = api_timestamp // 1000
```

### 19. Docstrings for All Public Functions and Methods

Use Google-style or NumPy-style docstrings consistently.

```python
def calculate_compound_interest(
    principal: Decimal,
    rate: Decimal,
    periods: int,
    compounds_per_period: int = 1,
) -> Decimal:
    """Calculate compound interest for a given principal.

    Uses the standard compound interest formula:
    A = P(1 + r/n)^(nt)

    Args:
        principal: The initial investment amount. Must be positive.
        rate: The annual interest rate as a decimal (e.g., 0.05 for 5%).
        periods: The number of years.
        compounds_per_period: Number of times interest compounds per year.
            Defaults to 1 (annual compounding).

    Returns:
        The final amount after compound interest, rounded to 2 decimal places.

    Raises:
        ValueError: If principal is negative or rate is negative.

    Examples:
        >>> calculate_compound_interest(Decimal("1000"), Decimal("0.05"), 10)
        Decimal('1628.89')
    """
    if principal < 0:
        raise ValueError("Principal must be non-negative")
    if rate < 0:
        raise ValueError("Rate must be non-negative")
    
    amount = principal * (1 + rate / compounds_per_period) ** (compounds_per_period * periods)
    return amount.quantize(Decimal("0.01"))
```

---

## Collections & Resources

### 20. Effective Use of Collections

Choose the appropriate collection type for the task.

```python
from collections import defaultdict, Counter, deque
from typing import Set

# Use set for membership testing and uniqueness
seen_ids: Set[str] = set()
if user_id in seen_ids:
    raise DuplicateError("User already processed")
seen_ids.add(user_id)

# Use defaultdict to avoid key existence checks
user_scores: defaultdict[str, int] = defaultdict(int)
user_scores[user_id] += points

# Use Counter for counting occurrences
word_counts = Counter(words)
most_common = word_counts.most_common(10)

# Use deque for efficient append/pop from both ends
recent_events: deque[Event] = deque(maxlen=100)
recent_events.append(new_event)

# Use tuple for fixed-size immutable sequences
coordinates: tuple[float, float] = (lat, lng)

# Use dict for key-value mappings
user_cache: dict[str, User] = {}
```

### 21. Resource Management

Use context managers (`with` statements) to properly close resources.

```python
# Avoid
f = open("file.txt")
try:
    data = f.read()
finally:
    f.close()

# Prefer
with open("file.txt") as f:
    data = f.read()

# Database connections
with database.connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()

# HTTP clients (async)
async with httpx.AsyncClient() as client:
    response = await client.get(url)

# Custom context managers
from contextlib import contextmanager

@contextmanager
def timed_operation(operation_name: str):
    """Context manager to log operation duration."""
    start = time.perf_counter()
    logger.info("Starting %s", operation_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Completed %s in %.3fs", operation_name, elapsed)

with timed_operation("data_processing"):
    process_data()
```

---

## Concurrency

### 22. Concurrent Programming

For async code, use `asyncio`. For threading, use proper synchronization primitives.

```python
import asyncio
from asyncio import Lock, Semaphore

# Async with proper synchronization
class RateLimiter:
    def __init__(self, max_requests: int, period_seconds: float):
        self._semaphore = Semaphore(max_requests)
        self._period = period_seconds

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        asyncio.create_task(self._release_after_period())

    async def _release_after_period(self) -> None:
        await asyncio.sleep(self._period)
        self._semaphore.release()

# Thread-safe cache with lock
from threading import Lock as ThreadLock

class ThreadSafeCache:
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._lock = ThreadLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value

# Concurrent task execution
async def fetch_all_users(user_ids: list[str]) -> list[User]:
    async with httpx.AsyncClient() as client:
        tasks = [fetch_user(client, uid) for uid in user_ids]
        return await asyncio.gather(*tasks)
```

---

## Testing & Quality

### 23. Unit Testing

Write comprehensive tests using `pytest`. Aim for high coverage of critical paths.

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestUserService:
    @pytest.fixture
    def mock_repo(self):
        return Mock(spec=UserRepository)

    @pytest.fixture
    def mock_email_service(self):
        return Mock(spec=EmailService)

    @pytest.fixture
    def user_service(self, mock_repo, mock_email_service):
        return UserService(repo=mock_repo, email_service=mock_email_service)

    def test_create_user_with_valid_data_succeeds(self, user_service, mock_repo):
        # Arrange
        request = CreateUserRequest(name="Alice", email="alice@example.com")
        mock_repo.find_by_email.return_value = None

        # Act
        user = user_service.create(request)

        # Assert
        assert user.name == "Alice"
        assert user.email == "alice@example.com"
        mock_repo.save.assert_called_once()

    def test_create_user_with_duplicate_email_raises(self, user_service, mock_repo):
        # Arrange
        existing_user = User(id="1", name="Bob", email="alice@example.com")
        mock_repo.find_by_email.return_value = existing_user
        request = CreateUserRequest(name="Alice", email="alice@example.com")

        # Act & Assert
        with pytest.raises(DuplicateEmailError) as exc_info:
            user_service.create(request)
        
        assert "alice@example.com" in str(exc_info.value)
        mock_repo.save.assert_not_called()

    @pytest.mark.parametrize("invalid_email", [
        "",
        "not-an-email",
        "@missing-local.com",
        "missing-domain@",
    ])
    def test_create_user_with_invalid_email_raises(
        self, user_service, invalid_email
    ):
        request = CreateUserRequest(name="Alice", email=invalid_email)
        
        with pytest.raises(ValidationError):
            user_service.create(request)

# Async test
@pytest.mark.asyncio
async def test_fetch_user_data_returns_user():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = Mock()
    mock_response.json.return_value = {"id": "1", "name": "Alice"}
    mock_client.get.return_value = mock_response

    result = await fetch_user_data(mock_client, "1")

    assert result.name == "Alice"
```

---

## Reuse & Style

### 24. Reuse Existing Libraries

Prefer well-maintained libraries over custom solutions.

```python
# HTTP requests: use httpx or aiohttp
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(url)

# Data validation: use pydantic
from pydantic import BaseModel, EmailStr

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr

# Retries: use tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def fetch_with_retry(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

# Date handling: use python-dateutil or pendulum
import pendulum

now = pendulum.now("UTC")
next_week = now.add(weeks=1)
```

### 25. Consistent Formatting

Use `ruff` or `black` for formatting, `mypy` for type checking.

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Async & Reactive

### 26. Non-Blocking & Reactive

For I/O-bound operations, use `async`/`await` with `asyncio` or frameworks like `FastAPI`.

```python
import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI()

class UserClient:
    def __init__(self, base_url: str):
        self._base_url = base_url

    async def fetch_user(self, user_id: str) -> UserData:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/users/{user_id}")
            response.raise_for_status()
            return UserData.model_validate(response.json())

    async def fetch_users(self, user_ids: list[str]) -> list[UserData]:
        async with httpx.AsyncClient() as client:
            tasks = [
                client.get(f"{self._base_url}/users/{uid}")
                for uid in user_ids
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            users = []
            for response in responses:
                if isinstance(response, Exception):
                    logger.warning("Failed to fetch user: %s", response)
                    continue
                users.append(UserData.model_validate(response.json()))
            return users

@app.get("/users/{user_id}")
async def get_user(user_id: str, client: UserClient = Depends(get_user_client)):
    try:
        return await client.fetch_user(user_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="User not found")
        raise
```

---

## Type Hints

### 27. Use Type Annotations Throughout

Leverage Python's type hints for clarity and static analysis.

```python
from typing import Optional, TypeVar, Generic
from collections.abc import Sequence, Callable, Awaitable

T = TypeVar("T")
E = TypeVar("E", bound=Exception)

# Generic types
class Result(Generic[T]):
    def __init__(self, value: T | None, error: Exception | None):
        self._value = value
        self._error = error

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(value=value, error=None)

    @classmethod
    def err(cls, error: Exception) -> "Result[T]":
        return cls(value=None, error=error)

# Function types
ProcessorFn = Callable[[Item], ProcessedItem]
AsyncProcessorFn = Callable[[Item], Awaitable[ProcessedItem]]

def process_items(
    items: Sequence[Item],
    processor: ProcessorFn,
    filter_fn: Optional[Callable[[Item], bool]] = None,
) -> list[ProcessedItem]:
    """Process a sequence of items with optional filtering.
    
    Args:
        items: The items to process.
        processor: Function to apply to each item.
        filter_fn: Optional predicate to filter items before processing.
    
    Returns:
        List of processed items.
    """
    if filter_fn is not None:
        items = [item for item in items if filter_fn(item)]
    
    return [processor(item) for item in items]

# TypedDict for structured dictionaries
from typing import TypedDict, NotRequired

class UserDict(TypedDict):
    id: str
    name: str
    email: str
    phone: NotRequired[str]
```

---

## Quick Reference Checklist

- [ ] Single responsibility per function/class
- [ ] Early returns instead of deep nesting
- [ ] Meaningful, descriptive names
- [ ] Specific exception handling
- [ ] Type hints on all public APIs
- [ ] Docstrings on all public functions
- [ ] Immutable data structures where possible
- [ ] Dependencies injected, not created
- [ ] Context managers for resources
- [ ] Async for I/O-bound operations
- [ ] Unit tests with good coverage
- [ ] Consistent formatting (ruff/black)
- [ ] No magic numbers
- [ ] No duplicate code
