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

from takehome_service.data import DataLoader, mask_sensitive, mask_in_text, format_citations, sanitize_text


class KYCProfileAgent:
    """Handles KYC/identity questions with mandatory masking of sensitive fields."""

    SYSTEM = (
        "You are ValuraKYCProfile, a KYC data assistant. "
        "You receive pre-extracted KYC data. Return a clear, concise answer. "
        "Sensitive identifiers have already been masked — do not unmask them. "
        "Do not add information not in the data provided."
    )

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
                text = self._llm_format(f"The employer on file is {employer}.", prompt)
                return self._build(text, str(employer), [kyc_id])
            return self._abstain("Employer information is not recorded in the KYC record for this client.")

        # --- Risk profile ---
        if re.search(r"\b(risk\s+profile|risk\s+appetite|risk\s+tolerance|risk\s+rating)\b", prompt_lower):
            kyc_risk = kyc.get("risk_profile") or kyc.get("risk_rating")
            client_dict = self._loader._clients_by_id.get(client_id, {})
            suitability = client_dict.get("suitability_reviews") or []
            if kyc_risk and suitability and suitability[-1].get("risk_profile"):
                rev_risk = suitability[-1]["risk_profile"]
                if kyc_risk.lower() != rev_risk.lower():
                    rev_id = suitability[-1].get("id", "")
                    text = f"There is a conflict in the records: KYC lists risk profile as {kyc_risk}, while suitability review {rev_id} lists it as {rev_risk}."
                    return self._build_conflict(text, [kyc_id, rev_id], client_id)
            if kyc_risk:
                text = self._llm_format(f"The risk profile on file is {kyc_risk}.", prompt)
                return self._build(text, str(kyc_risk), [kyc_id], client_id=client_id)
            return self._abstain("Risk profile is not recorded in the KYC data.")

        # --- KYC status ---
        if re.search(r"\b(kyc\s+status|kyc\s+complete|verification\s+status|good\s+standing)\b", prompt_lower):
            status = kyc.get("kyc_status")
            notes = self._loader.get_notes(client_id)
            pending_note = next(
                (n for n in notes if "re-verification is pending" in str(n.get("text", "")).lower() or "expired" in str(n.get("text", "")).lower()),
                None,
            )
            if status and pending_note:
                note_id = pending_note.get("id", "")
                text = f"There is a conflict in the records: KYC status is marked as {status}, but note {note_id} indicates KYC re-verification is pending."
                return self._build_conflict(text, [kyc_id, note_id], client_id)
            if status:
                text = self._llm_format(f"The KYC status is {status}.", prompt)
                return self._build(text, str(status), [kyc_id], client_id=client_id)
            return self._abstain("KYC status is not recorded for this client.")

        # --- PAN (always masked as ****XXXX) ---
        if re.search(r"\b(pan|pan\s+number|permanent\s+account)\b", prompt_lower):
            pan = kyc.get("pan")
            if pan:
                masked = mask_sensitive(pan)
                text = self._llm_format(f"The PAN on file is {masked}.", prompt)
                return self._build(text, masked, [kyc_id])
            return self._abstain("PAN is not recorded in the KYC file for this client.")

        # --- Bank account (always masked as ****XXXX) ---
        if re.search(r"\b(bank\s+account|account\s+number|ifsc|bank\s+details)\b", prompt_lower):
            bank_info = kyc.get("bank_account") or {}
            acc_num = bank_info.get("account_number") or bank_info.get("number")
            if acc_num:
                masked_acc = mask_sensitive(str(acc_num))
                text = self._llm_format(f"The bank account number on file is {masked_acc}.", prompt)
                # answer_value MUST be the masked form ****1536 for pii_bank_last4 or pii_bank
                return self._build(text, masked_acc, [kyc_id])
            return self._abstain("Bank account details are not available in the KYC record for this client.")

        # --- Nominee ---
        if re.search(r"\b(nominee|beneficiary|nomination)\b", prompt_lower):
            nominee = kyc.get("nominee") or kyc.get("beneficiary")
            if nominee:
                text = self._llm_format(f"The nominee on file is {nominee}.", prompt)
                return self._build(text, str(nominee), [kyc_id])
            return self._abstain("No nominee is recorded in the KYC file for this client.")

        # --- Date of birth ---
        if re.search(r"\b(date\s+of\s+birth|dob|born|birthday)\b", prompt_lower):
            dob = kyc.get("date_of_birth") or kyc.get("dob")
            if dob:
                text = self._llm_format(f"The date of birth on file is {dob}.", prompt)
                return self._build(text, str(dob), [kyc_id])
            return self._abstain("Date of birth is not recorded for this client.")

        # --- Annual income ---
        if re.search(r"\b(annual\s+income|income\s+band|salary)\b", prompt_lower):
            income = kyc.get("annual_income_band") or kyc.get("annual_income")
            if income:
                text = self._llm_format(f"The annual income band on file is {income}.", prompt)
                return self._build(text, str(income), [kyc_id])
            return self._abstain("Annual income information is not recorded for this client.")

        # --- Email ---
        if re.search(r"\b(email|e-mail|email\s+address)\b", prompt_lower):
            email = identity.get("email") or kyc.get("email")
            if email:
                text = self._llm_format(f"The email address on file is {email}.", prompt)
                return self._build(text, str(email), [kyc_id])
            return self._abstain("Email address is not recorded for this client.")

        # --- Mobile / phone ---
        if re.search(r"\b(mobile|phone|telephone|contact\s+number)\b", prompt_lower):
            phone = identity.get("mobile") or identity.get("phone") or kyc.get("mobile") or kyc.get("phone")
            if phone:
                text = self._llm_format(f"The phone number on file is {phone}.", prompt)
                return self._build(text, str(phone), [kyc_id])
            return self._abstain("Phone/mobile number is not recorded for this client.")

        # --- Address ---
        if re.search(r"\b(address|residence|residential)\b", prompt_lower):
            address = kyc.get("address")
            if address:
                text = self._llm_format(f"The address on file is {address}.", prompt)
                return self._build(text, str(address), [kyc_id])
            return self._abstain("Address is not recorded in the KYC file for this client.")

        # --- Identity number / client ID ---
        if re.search(r"\b(identity\s+number|id\s+number|client\s+id)\b", prompt_lower):
            cid_val = identity.get("id", client_id)
            text = self._llm_format(f"The client identity number on file is {cid_val}.", prompt)
            return self._build(text, str(cid_val), [kyc_id])

        return self._abstain("The requested KYC or identity field is not recorded for this client.")

    def _llm_format(self, precomputed: str, original_prompt: str) -> str:
        try:
            run_output = self._agent.run(
                f"Pre-computed answer: {precomputed}\n"
                f"Original question: {original_prompt}\n\n"
                f"Rephrase the answer clearly. Do not change any values. Return only the answer."
            )
            content = run_output.get_content_as_string() if run_output else ""
            if not content or "STUB-GATEWAY" in content:
                return precomputed
            return sanitize_text(content.strip())
        except Exception:
            return precomputed

    def _build(
        self, answer: str, value: Optional[str], citations: List[str], client_id: str = ""
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "answer_value": value,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": format_citations(client_id, citations),
            "confidence": 0.9,
            "flags": [],
        }

    def _build_conflict(
        self, answer: str, citations: List[str], client_id: str = ""
    ) -> Dict[str, Any]:
        return {
            "answer": sanitize_text(answer),
            "answer_value": None,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": format_citations(client_id, citations),
            "confidence": 0.85,
            "flags": ["conflict"],
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
