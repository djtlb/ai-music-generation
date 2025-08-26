#!/bin/bash
# Test script for Stripe integration

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing Stripe Integration...${NC}"

# Test variables
TEST_EMAIL="test@example.com"
TEST_NAME="Test User"

# 1. Test create-customer endpoint
echo -e "\n${YELLOW}1. Testing create-customer endpoint...${NC}"
CUSTOMER_RESPONSE=$(curl -s -X POST 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1/create-customer' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTc4NTYsImV4cCI6MjA3MDg3Mzg1Nn0.gKtvk5fiqjBToyy9w0_-DF4EmZdBQwUJCpY07XM7QJ0' \
  -d "{\"email\": \"$TEST_EMAIL\", \"name\": \"$TEST_NAME\"}")

echo "Response: $CUSTOMER_RESPONSE"

# Check if customer creation was successful
if echo "$CUSTOMER_RESPONSE" | grep -q "\"success\":true"; then
  echo -e "${GREEN}✓ Customer created successfully${NC}"
  CUSTOMER_ID=$(echo "$CUSTOMER_RESPONSE" | grep -o '"customerId":"[^"]*' | sed 's/"customerId":"//')
  echo "Customer ID: $CUSTOMER_ID"
else
  echo -e "${RED}✗ Failed to create customer${NC}"
  exit 1
fi

# 2. Test create-subscription endpoint
echo -e "\n${YELLOW}2. Testing create-subscription endpoint...${NC}"
SUBSCRIPTION_RESPONSE=$(curl -s -X POST 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1/create-subscription' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTc4NTYsImV4cCI6MjA3MDg3Mzg1Nn0.gKtvk5fiqjBToyy9w0_-DF4EmZdBQwUJCpY07XM7QJ0' \
  -d "{\"customerId\": \"$CUSTOMER_ID\", \"priceId\": \"price_1RzsuQP2t49TLtZwBGCR42bt\", \"addressInfo\": {\"line1\": \"123 Main St\", \"city\": \"San Francisco\", \"state\": \"CA\", \"postal_code\": \"94111\", \"country\": \"US\"}}")

echo "Response: $SUBSCRIPTION_RESPONSE"

# Check if subscription creation was successful
if echo "$SUBSCRIPTION_RESPONSE" | grep -q "\"success\":true"; then
  echo -e "${GREEN}✓ Subscription created successfully${NC}"
  SUBSCRIPTION_ID=$(echo "$SUBSCRIPTION_RESPONSE" | grep -o '"subscriptionId":"[^"]*' | sed 's/"subscriptionId":"//')
  echo "Subscription ID: $SUBSCRIPTION_ID"
else
  echo -e "${RED}✗ Failed to create subscription${NC}"
  # Don't exit, continue with payment test
fi

# 3. Test create-payment endpoint
echo -e "\n${YELLOW}3. Testing create-payment endpoint...${NC}"
PAYMENT_RESPONSE=$(curl -s -X POST 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1/create-payment' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTc4NTYsImV4cCI6MjA3MDg3Mzg1Nn0.gKtvk5fiqjBToyy9w0_-DF4EmZdBQwUJCpY07XM7QJ0' \
  -d "{\"customerId\": \"$CUSTOMER_ID\", \"amount\": 1999, \"currency\": \"usd\", \"description\": \"Test payment\"}")

echo "Response: $PAYMENT_RESPONSE"

# Check if payment creation was successful
if echo "$PAYMENT_RESPONSE" | grep -q "\"success\":true"; then
  echo -e "${GREEN}✓ Payment created successfully${NC}"
  PAYMENT_INTENT_ID=$(echo "$PAYMENT_RESPONSE" | grep -o '"paymentIntentId":"[^"]*' | sed 's/"paymentIntentId":"//')
  echo "Payment Intent ID: $PAYMENT_INTENT_ID"
else
  echo -e "${RED}✗ Failed to create payment${NC}"
fi

echo -e "\n${GREEN}Stripe integration test complete${NC}"
