# Proposed IntentScope annotation guidelines

These project rules are drafts for human validation. Use the 77 names in `categories.json` as the allowed taxonomy. Read the original query before seeing its source label. Choose the narrowest intent supported by the words in the query, based on the user's requested action and transaction type. Write a brief rationale that quotes or paraphrases the evidence.

If two intents remain plausible, mark **ambiguous** and name the competing labels in the rationale. If the required transaction state or product is missing, mark **needs context** and state what detail would resolve it. Confidence is 1 (guess) to 5 (clear wording); it measures the reviewer's certainty, not model confidence.

A source-label disagreement is a **review candidate**. Record the alternate label and why it fits; do not call the original wrong without independent adjudication. A data defect concerns the source record, such as missing text, an unusable label, or a verified transcription problem. A model error requires a saved model prediction and a human-validated reference label. This workflow does not inspect test predictions or decide model errors.

## Proposed decision rules for close pairs

| Labels | Proposed distinction | Example wording |
| --- | --- | --- |
| `why_verify_identity` / `verify_my_identity` | Choose the first for a reason or policy question; the second for how to complete verification. | “Why do you need my ID?” / “How do I verify my ID?” |
| `card_arrival` / `card_delivery_estimate` | Choose arrival for a card apparently overdue or missing; estimate for a requested delivery date. | “My card still has not arrived” / “When should my card arrive?” |
| `pending_card_payment` / `reverted_card_payment?` | Choose pending for an unsettled payment; reverted for a payment that has reversed or returned. | “The card payment is pending” / “Why did that card payment reverse?” |
| `declined_transfer` / `failed_transfer` | Use declined when rejection is explicit; use failed when the attempt did not complete but no rejection is stated. Ask for context if the wording does not distinguish them. | “My transfer was declined” / “My transfer failed to go through.” |
| `cash_withdrawal_charge` / `wrong_exchange_rate_for_cash_withdrawal` | Distinguish a fee from an exchange-rate complaint for a cash withdrawal. | “Why was there an ATM fee?” / “The ATM conversion rate was wrong.” |

These examples are illustrative, not reviewed dataset records. A human domain reviewer should approve or revise the rules before using them as authoritative annotation policy.
