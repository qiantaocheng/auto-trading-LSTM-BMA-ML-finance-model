#!/usr/bin/env python3
"""
🔐 Secure Configuration Manager
Provides encrypted storage and secure access to sensitive configuration data
"""

import os
import logging
import base64
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from pathlib import Path

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logging.warning("keyring not available - using file-based key storage (less secure)")


class SecureConfigManager:
    """Secure configuration manager with encryption for sensitive data"""
    
    def __init__(self, service_name: str = "autotrader"):
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        self._cipher = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption key safely"""
        try:
            key = self._get_or_create_encryption_key()
            self._cipher = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise RuntimeError("Cannot initialize secure config without encryption")
    
    def _get_or_create_encryption_key(self) -> str:
        """Get or create encryption key with fallback methods"""
        
        # Method 1: Try system keyring (most secure)
        if KEYRING_AVAILABLE:
            try:
                key = keyring.get_password(self.service_name, "config_encryption_key")
                if key:
                    return key
                
                # Generate new key and store in keyring
                new_key = Fernet.generate_key().decode()
                keyring.set_password(self.service_name, "config_encryption_key", new_key)
                self.logger.info("Created new encryption key in system keyring")
                return new_key
                
            except Exception as e:
                self.logger.warning(f"Keyring unavailable: {e}")
        
        # Method 2: Environment variable (less secure)
        env_key = os.getenv('AUTOTRADER_CONFIG_KEY')
        if env_key:
            return env_key
        
        # Method 3: File-based storage (least secure, but functional)
        key_file = Path.home() / f".{self.service_name}" / "config.key"
        key_file.parent.mkdir(exist_ok=True, mode=0o700)
        
        if key_file.exists():
            with open(key_file, 'r') as f:
                return f.read().strip()
        else:
            new_key = Fernet.generate_key().decode()
            with open(key_file, 'w') as f:
                f.write(new_key)
            key_file.chmod(0o600)  # Owner read/write only
            self.logger.warning(f"Created encryption key file: {key_file}")
            return new_key
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a string value"""
        if not self._cipher:
            raise RuntimeError("Encryption not initialized")
        
        encrypted_bytes = self._cipher.encrypt(value.encode())
        return base64.b64encode(encrypted_bytes).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted string value"""
        if not self._cipher:
            raise RuntimeError("Encryption not initialized")
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted_bytes = self._cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt value: {e}")
    
    def store_sensitive_config(self, key: str, value: str) -> str:
        """Store sensitive configuration value and return encrypted version"""
        encrypted = self.encrypt_value(value)
        
        # Clear the original value from memory if it was in environment
        env_key = key.upper().replace('.', '_')
        if env_key in os.environ:
            del os.environ[env_key]
            self.logger.debug(f"Cleared {env_key} from environment")
        
        return encrypted
    
    def get_sensitive_config(self, encrypted_value: str) -> str:
        """Retrieve and decrypt sensitive configuration value"""
        try:
            return self.decrypt_value(encrypted_value)
        except ValueError as e:
            self.logger.error(f"Failed to decrypt config value: {e}")
            raise
    
    def secure_log_filter(self, message: str, sensitive_patterns: list = None) -> str:
        """Filter sensitive information from log messages"""
        if sensitive_patterns is None:
            sensitive_patterns = [
                r'api[_-]?key[\'\":\s=]+[\'\"]\w+[\'\""]',
                r'password[\'\":\s=]+[\'\"]\w+[\'\""]',
                r'secret[\'\":\s=]+[\'\"]\w+[\'\""]',
                r'token[\'\":\s=]+[\'\"]\w+[\'\""]',
            ]
        
        sanitized = message
        import re
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized


def get_secure_config_manager() -> SecureConfigManager:
    """Get singleton secure config manager"""
    if not hasattr(get_secure_config_manager, '_instance'):
        get_secure_config_manager._instance = SecureConfigManager()
    return get_secure_config_manager._instance