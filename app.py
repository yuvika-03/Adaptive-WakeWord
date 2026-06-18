#!/usr/bin/env python3
"""
Adaptive WakeWord: Personalized Speaker-Aware Voice Authentication
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from inference.mfcc_authenticator import MFCCAuthenticator
from inference.enroll_mfcc import MFCCEnroller

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive WakeWord - Personalized Voice Authentication"
    )
    parser.add_argument(
        "--mode",
        choices=["enroll", "authenticate", "demo"],
        default="demo",
        help="Operation mode"
    )
    parser.add_argument(
        "--username",
        type=str,
        default="test_user",
        help="Username for enrollment"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎤 Adaptive WakeWord - Personalized Voice Authentication")
    print("=" * 60)
    
    if args.mode == "enroll":
        print(f"\n👤 Enrolling user: {args.username}")
        enroller = MFCCEnroller()
        enroller.enroll_speaker(args.username)
        
    elif args.mode == "authenticate":
        print("\n🔐 Starting authentication mode...")
        authenticator = MFCCAuthenticator()
        authenticator.start()
        
    elif args.mode == "demo":
        print("\n🎯 Quick Demo Guide:")
        print("=" * 60)
        print("\n1. Enroll a user:")
        print("   python app.py --mode enroll --username your_name")
        print("\n2. Run authentication:")
        print("   python app.py --mode authenticate")
        print("\n3. Speak your wake word and watch the magic! ✨")

if __name__ == "__main__":
    main()