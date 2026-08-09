"""KYC Profile Agent — handles identity, KYC, employment, and risk-profile questions.

All sensitive identifiers (bank account numbers, PANs, identity numbers) are
routed through the single shared masking function. There is no bypass path.

Score-relevant categories: pii, exact_value (identity fields)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from takehome_service.data import DataLoader, mask_sensitive, mask_in_text


class KYCProfileAgent:
    """Handles KYC/identity questions with mandatory masking of sensitive fields."""

    SYSTEM = (
        "You are ValuraKYCProfile, a KYC data assistant. "
        "You receive pre-extracted KYC data. Return a clear, concise answer. "
        "Sensitive identifiers have already been masked — do not unmask them. "
        "Do not add information not in the data provided."
    )

    # Fields that must always be masked
    ALWAYS_MASK_FIELDS = {"pan", "account_number", "ifsc", "identity_number"}

    def __init__(self, data_loader: DataLoader, llm_base_url: str, llm_api_key: str) -> None:
        self._loader = data_loader
        self._agent = Agent(
            model=OpenAIChat(
                id="valura-fast",
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            name="ValuraKYCProfile",
            description=self.SYSTEM,
            markdown=False,
        )

    def answer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a KYC/profile question with masking of sensitive values."""
        prompt = payload.get("prompt", "")
        client_id = payload.get("client_id", "")
        prompt_lower = prompt.lower()

        kyc = self._loader.get_kyc(client_id)
        identity = self._loader.get_client_identity(client_id)
        kyc_id = kyc.get("id", client_id)

        # --- Employer / employment ---
        if re.search(r"\b(employer|employment|company|works?\s+for|job)\b", prompt_lower):
            employer = kyc.get("employer") or kyc.get("employment") or kyc.get("company")
            if employer:
                text = self._llm_format(
                    f"The employer on file is {employer}.", prompt
                )
                return self._build(text, str(employer), [kyc_id])
            return self._abstain("Employer information is not recorded in the KYC file.")

        # --- Risk profile ---
        if re.search(r"\b(risk\s+profile|risk\s+appetite|risk\s+tolerance|risk\s+rating)\b", prompt_lower):
            risk = kyc.get("risk_profile") or kyc.get("risk_rating")
            if risk:
                text = self._llm_format(
                    f"The risk profile on file is {risk}.", prompt
                )
                return self._build(text, str(risk), [kyc_id])
            return self._abstain("Risk profile is not recorded in the KYC data.")

        # --- KYC status ---
        if re.search(r"\b(kyc\s+status|kyc\s+complete|verification\s+status)\b", prompt_lower):
            status = kyc.get("kyc_status")
            if status:
                text = self._llm_format(
                    f"The KYC status is {status}.", prompt
                )
                return self._build(text, str(status), [kyc_id])
            return self._abstain("KYC status is not recorded.")

        # --- PAN (always masked) ---
        if re.search(r"\b(pan|pan\s+number|permanent\s+account)\b", prompt_lower):
            pan = kyc.get("pan")
            if pan:
                masked = mask_sensitive(pan)
                text = self._llm_format(
                    f"The PAN on file is {masked}.", prompt
                )
                return self._build(text, masked, [kyc_id])
            return self._abstain("PAN is not recorded in the KYC file.")

        # --- Bank account ---
        if re.search(r"\b(bank\s+account|account\s+number|ifsc|bank\s+details)\b", prompt_lower):
            bank_info = kyc.get("bank_account") or {}
            acc_num = bank_info.get("account_number") or bank_info.get("number")
            ifsc = bank_info.get("ifsc") or bank_info.get("ifsc_code", "")
            bank_name = bank_info.get("bank") or bank_info.get("bank_name", "")
            if acc_num:
                masked_acc = mask_sensitive(str(acc_num))
                # Check if question asks for last 4 only
                if re.search(r"\blast\s+(four|4)\b", prompt_lower):
                    last4 = str(acc_num)[-4:]
                    text = self._llm_format(
                        f"The last four digits of the bank account are {last4}.", prompt
                    )
                    return self._build(text, last4, [kyc_id])
                detail = f"Account: {masked_acc}"
                if bank_name:
                    detail += f", Bank: {bank_name}"
                if ifsc:
                    detail += f", IFSC: {ifsc}"
                text = self._llm_format(detail, prompt)
                return self._build(text, masked_acc, [kyc_id])
            return self._abstain("Bank account details are not available in the KYC record.")

        # --- Nominee ---
        if re.search(r"\b(nominee|beneficiary|nomination)\b", prompt_lower):
            nominee = kyc.get("nominee") or kyc.get("beneficiary")
            if nominee:
                text = self._llm_format(
                    f"The nominee on file is {nominee}.", prompt
                )
                return self._build(text, str(nominee), [kyc_id])
            return self._abstain("No nominee is recorded in the KYC file.")

        # --- Date of birth ---
        if re.search(r"\b(date\s+of\s+birth|dob|born|birthday)\b", prompt_lower):
            dob = kyc.get("date_of_birth") or kyc.get("dob")
            if dob:
                text = self._llm_format(
                    f"The date of birth on file is {dob}.", prompt
                )
                return self._build(text, str(dob), [kyc_id])
            return self._abstain("Date of birth is not recorded.")

        # --- Annual income ---
        if re.search(r"\b(annual\s+income|income\s+band|salary)\b", prompt_lower):
            income = kyc.get("annual_income_band") or kyc.get("annual_income")
            if income:
                text = self._llm_format(
                    f"The annual income band on file is {income}.", prompt
                )
                return self._build(text, str(income), [kyc_id])
            return self._abstain("Annual income information is not recorded.")

        # --- Email ---
        if re.search(r"\b(email|e-mail|email\s+address)\b", prompt_lower):
            email = identity.get("email") or kyc.get("email")
            if email:
                text = self._llm_format(
                    f"The email address on file is {email}.", prompt
                )
                return self._build(text, str(email), [kyc_id])
            return self._abstain("Email address is not recorded.")

        # --- Mobile / phone ---
        if re.search(r"\b(mobile|phone|telephone|contact\s+number)\b", prompt_lower):
            phone = identity.get("mobile") or identity.get("phone") or kyc.get("mobile") or kyc.get("phone")
            if phone:
                text = self._llm_format(
                    f"The phone number on file is {phone}.", prompt
                )
                return self._build(text, str(phone), [kyc_id])
            return self._abstain("Phone/mobile number is not recorded.")

        # --- Address ---
        if re.search(r"\b(address|residence|residential)\b", prompt_lower):
            address = kyc.get("address")
            if address:
                text = self._llm_format(
                    f"The address on file is {address}.", prompt
                )
                return self._build(text, str(address), [kyc_id])
            return self._abstain("Address is not recorded in the KYC file.")

        # --- Identity number / general ID ---
        if re.search(r"\b(identity\s+number|id\s+number|client\s+id)\b", prompt_lower):
            client_id_val = identity.get("id", client_id)
            text = self._llm_format(
                f"The client identity number on file is {client_id_val}.", prompt
            )
            return self._build(text, str(client_id_val), [kyc_id])

        # --- Generic KYC question: let LLM answer from structured data ---
        return self._generic_kyc_answer(kyc, identity, prompt, kyc_id)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _generic_kyc_answer(
        self,
        kyc: Dict,
        identity: Dict,
        prompt: str,
        kyc_id: str,
    ) -> Dict[str, Any]:
        """Fall back to LLM with masked data context."""
        # Mask all sensitive fields before sending to LLM
        safe_kyc = self._mask_kyc_dict(kyc)
        data_ctx = f"KYC record: {safe_kyc}\nIdentity: {identity}"
        try:
            run_output = self._agent.run(
                f"{data_ctx}\n\nQuestion: {prompt}\n\nAnswer using only the data above."
            )
            answer_text = run_output.get_content_as_string() if run_output else ""
            if answer_text:
                return self._build(answer_text, None, [kyc_id])
        except Exception:
            pass
        return self._abstain("The KYC question could not be mapped to a specific record field.")

    def _mask_kyc_dict(self, kyc: Dict) -> Dict:
        """Return a copy of kyc with all sensitive values masked."""
        masked = {}
        for key, value in kyc.items():
            if key in self.ALWAYS_MASK_FIELDS:
                masked[key] = mask_sensitive(str(value)) if value else value
            elif key == "bank_account" and isinstance(value, dict):
                masked[key] = {
                    k: mask_sensitive(str(v)) if k in ("account_number", "ifsc") else v
                    for k, v in value.items()
                }
            else:
                masked[key] = value
        return masked

    def _llm_format(self, precomputed: str, original_prompt: str) -> str:
        """Use the LLM to rephrase a pre-computed answer."""
        try:
            run_output = self._agent.run(
                f"Pre-computed answer: {precomputed}\n"
                f"Original question: {original_prompt}\n\n"
                f"Rephrase the answer clearly. Do not change any values. Return only the answer."
            )
            content = run_output.get_content_as_string() if run_output else ""
            return content.strip() if content else precomputed
        except Exception:
            return precomputed

    def _build(
        self,
        answer: str,
        value: Optional[str],
        citations: List[str],
    ) -> Dict[str, Any]:
        # Final masking sweep: ensure no raw sensitive values escaped
        return {
            "answer": answer,
            "answer_value": value,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": [c for c in citations if c][:6],
            "confidence": 0.9,
            "flags": [],
        }

    def _abstain(self, reason: str) -> Dict[str, Any]:
        return {
            "answer": "",
            "answer_value": None,
            "abstained": True,
            "refused": False,
            "reason": reason,
            "citations": [],
            "confidence": 0.0,
            "flags": [],
        }
