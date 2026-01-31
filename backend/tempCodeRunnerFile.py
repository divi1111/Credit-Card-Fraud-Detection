class TransactionInput(BaseModel):
    """
    Schema for incoming transaction data from frontend.
    All fields match the expected input from the HTML form.
    """
    step: int = Field(..., ge=1, description="Time step of the transaction")
    types: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"] = Field(
        ..., description="Type of transaction"
    )
    amount: float = Field(..., ge=0, description="Transaction amount")
    oldbalanceorig: float = Field(..., ge=0, description="Original balance before transaction")
    newbalanceorig: float = Field(..., ge=0, description="New balance after transaction")
    oldbalancedest: float = Field(..., ge=0, description="Destination original balance")
    newbalancedest: float = Field(..., ge=0, description="Destination new balance")
    isflaggedfraud: int = Field(..., ge=0, le=1, description="System fraud flag (0 or 1)")