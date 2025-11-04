#!/bin/bash
# AITA API Test Script
# Tests all API endpoints with various scenarios

echo "=================================="
echo "AITA API Test Suite"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_URL="http://127.0.0.1:8000"

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test endpoint
test_endpoint() {
    local name=$1
    local endpoint=$2
    local expected_status=$3

    echo -n "Testing $name... "

    status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$endpoint")

    if [ "$status" -eq "$expected_status" ]; then
        echo -e "${GREEN}PASS${NC} (Status: $status)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}FAIL${NC} (Expected: $expected_status, Got: $status)"
        ((TESTS_FAILED++))
    fi
}

echo "=== Basic Endpoints ==="
test_endpoint "Root endpoint" "/" 200
test_endpoint "Health check" "/health" 200
test_endpoint "API docs" "/docs" 200
test_endpoint "OpenAPI schema" "/openapi.json" 200
echo ""

echo "=== Valid Symbol Analysis ==="
test_endpoint "NVDA analysis" "/analyze?symbol=NVDA" 200
test_endpoint "AAPL analysis" "/analyze?symbol=AAPL" 200
test_endpoint "TSLA analysis" "/analyze?symbol=TSLA" 200
test_endpoint "MSFT analysis" "/analyze?symbol=MSFT" 200
test_endpoint "Default (AVGO) analysis" "/analyze" 200
echo ""

echo "=== Error Handling ==="
test_endpoint "Invalid symbol" "/analyze?symbol=INVALID_SYMBOL" 404
test_endpoint "Empty symbol" "/analyze?symbol=" 404
echo ""

echo "=== Detailed Pattern Detection Test ==="
echo "Analyzing NVDA for patterns..."
response=$(curl -s "$BASE_URL/analyze?symbol=NVDA")
patterns=$(echo "$response" | python -c "import sys, json; d=json.load(sys.stdin); print(len(d['summary']['patterns']))" 2>/dev/null)

if [ ! -z "$patterns" ] && [ "$patterns" -gt 0 ]; then
    echo -e "${GREEN}PASS${NC} - Detected $patterns pattern(s)"
    echo "$response" | python -c "import sys, json; d=json.load(sys.stdin); [print(f\"  - {p['name']}: {p['score']*100:.0f}% confidence\") for p in d['summary']['patterns']]" 2>/dev/null
    ((TESTS_PASSED++))
else
    echo -e "${RED}FAIL${NC} - No patterns detected"
    ((TESTS_FAILED++))
fi
echo ""

echo "=== Trade Plan Validation ==="
echo "Checking NVDA trade plan..."
trade_plan=$(curl -s "$BASE_URL/analyze?symbol=NVDA" | python -c "
import sys, json
d = json.load(sys.stdin)
plan = d['summary']['plan']
print(f\"Direction: {plan['direction']}\")
print(f\"Entry: \${plan['entry']}\")
print(f\"Stop: \${plan['stop']}\")
print(f\"Target 1: \${plan['targets'][0]}\")
print(f\"Risk/Reward: {plan['risk_reward']}\")
print(f\"Options: {plan['options']['structure']}\")
" 2>/dev/null)

if [ ! -z "$trade_plan" ]; then
    echo -e "${GREEN}PASS${NC}"
    echo "$trade_plan" | sed 's/^/  /'
    ((TESTS_PASSED++))
else
    echo -e "${RED}FAIL${NC} - Trade plan validation failed"
    ((TESTS_FAILED++))
fi
echo ""

echo "=== Performance Test ==="
echo "Running 5 sequential analyses..."
start_time=$(date +%s)
for symbol in NVDA AAPL TSLA MSFT GOOGL; do
    curl -s "$BASE_URL/analyze?symbol=$symbol" > /dev/null
    echo -n "."
done
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo ""
if [ "$elapsed" -lt 30 ]; then
    echo -e "${GREEN}PASS${NC} - Completed in ${elapsed}s"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}SLOW${NC} - Completed in ${elapsed}s (expected < 30s)"
    ((TESTS_PASSED++))
fi
echo ""

echo "=================================="
echo "Test Results"
echo "=================================="
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo "Total: $((TESTS_PASSED + TESTS_FAILED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
