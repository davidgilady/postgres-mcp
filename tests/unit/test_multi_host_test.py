import os
from unittest.mock import patch

import pytest

import postgres_mcp.server as server_module
from postgres_mcp.models import AccessMode
from postgres_mcp.models import HostConfig
from postgres_mcp.server import create_database_url_from_config
from postgres_mcp.server import get_service
from postgres_mcp.server import parse_host_configs_from_env
from postgres_mcp.server import resolve_host_config

# --- parse_host_configs_from_env ---


class TestParseHostConfigsFromEnv:
    def test_no_env_vars_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            assert parse_host_configs_from_env() == {}

    def test_single_host_parsed(self):
        env = {
            "DATABASES__1__HOST": "db1.example.com",
            "DATABASES__1__PORT": "5433",
            "DATABASES__1__USERNAME": "user1",
            "DATABASES__1__PASSWORD": "pass1",
        }
        with patch.dict(os.environ, env, clear=True):
            configs = parse_host_configs_from_env()
            assert len(configs) == 1
            config = configs["db1.example.com"]
            assert config.host == "db1.example.com"
            assert config.port == 5433
            assert config.username == "user1"
            assert config.password == "pass1"

    def test_multiple_hosts_parsed(self):
        env = {
            "DATABASES__1__HOST": "db1.example.com",
            "DATABASES__1__PORT": "5432",
            "DATABASES__1__USERNAME": "user1",
            "DATABASES__1__PASSWORD": "pass1",
            "DATABASES__2__HOST": "db2.example.com",
            "DATABASES__2__PORT": "5433",
            "DATABASES__2__USERNAME": "user2",
            "DATABASES__2__PASSWORD": "pass2",
        }
        with patch.dict(os.environ, env, clear=True):
            configs = parse_host_configs_from_env()
            assert len(configs) == 2
            assert "db1.example.com" in configs
            assert "db2.example.com" in configs
            assert configs["db2.example.com"].port == 5433

    def test_default_port_when_omitted(self):
        env = {
            "DATABASES__1__HOST": "db1.example.com",
            "DATABASES__1__USERNAME": "user1",
            "DATABASES__1__PASSWORD": "pass1",
        }
        with patch.dict(os.environ, env, clear=True):
            configs = parse_host_configs_from_env()
            assert configs["db1.example.com"].port == 5432

    def test_missing_username_raises(self):
        env = {
            "DATABASES__1__HOST": "db1.example.com",
            "DATABASES__1__PASSWORD": "pass1",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="DATABASES__1__USERNAME and DATABASES__1__PASSWORD must both be set"):
                parse_host_configs_from_env()

    def test_missing_password_raises(self):
        env = {
            "DATABASES__1__HOST": "db1.example.com",
            "DATABASES__1__USERNAME": "user1",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="DATABASES__1__USERNAME and DATABASES__1__PASSWORD must both be set"):
                parse_host_configs_from_env()

    def test_non_numeric_indices_supported(self):
        env = {
            "DATABASES__prod__HOST": "prod.example.com",
            "DATABASES__prod__USERNAME": "admin",
            "DATABASES__prod__PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            configs = parse_host_configs_from_env()
            assert "prod.example.com" in configs

    def test_unrelated_env_vars_ignored(self):
        env = {
            "DATABASES__1__HOST": "db1.example.com",
            "DATABASES__1__USERNAME": "user1",
            "DATABASES__1__PASSWORD": "pass1",
            "SOME_OTHER_VAR": "value",
            "DATABASES__1__EXTRA": "ignored",
        }
        with patch.dict(os.environ, env, clear=True):
            configs = parse_host_configs_from_env()
            assert len(configs) == 1


# --- resolve_host_config ---


class TestResolveHostConfig:
    CONFIG_A = HostConfig(host="a.example.com", port=5432, username="ua", password="pa")
    CONFIG_B = HostConfig(host="b.example.com", port=5433, username="ub", password="pb")

    def test_single_host_no_host_param_returns_it(self):
        with patch.object(server_module, "host_configs", {"a.example.com": self.CONFIG_A}):
            assert resolve_host_config(None) == self.CONFIG_A

    def test_single_host_with_matching_host_param(self):
        with patch.object(server_module, "host_configs", {"a.example.com": self.CONFIG_A}):
            assert resolve_host_config("a.example.com") == self.CONFIG_A

    def test_single_host_with_wrong_host_param_raises(self):
        with patch.object(server_module, "host_configs", {"a.example.com": self.CONFIG_A}):
            with pytest.raises(ValueError, match="No configuration found for host 'unknown'"):
                resolve_host_config("unknown")

    def test_multiple_hosts_no_host_param_raises(self):
        configs = {"a.example.com": self.CONFIG_A, "b.example.com": self.CONFIG_B}
        with patch.object(server_module, "host_configs", configs):
            with pytest.raises(ValueError, match="'host' parameter is required when multiple hosts are configured"):
                resolve_host_config(None)

    def test_multiple_hosts_with_host_param(self):
        configs = {"a.example.com": self.CONFIG_A, "b.example.com": self.CONFIG_B}
        with patch.object(server_module, "host_configs", configs):
            assert resolve_host_config("b.example.com") == self.CONFIG_B

    def test_no_hosts_configured_raises(self):
        with patch.object(server_module, "host_configs", {}):
            with pytest.raises(ValueError, match="No database host configured"):
                resolve_host_config(None)


# --- create_database_url_from_config ---


class TestCreateDatabaseUrlFromConfig:
    def test_url_format(self):
        config = HostConfig(host="myhost", port=5432, username="myuser", password="mypass")
        url = create_database_url_from_config(config, "mydb")
        assert url == "postgresql://myuser:mypass@myhost:5432/mydb"

    def test_url_with_custom_port(self):
        config = HostConfig(host="myhost", port=5433, username="u", password="p")
        url = create_database_url_from_config(config, "testdb")
        assert url == "postgresql://u:p@myhost:5433/testdb"


# --- get_service ---


class TestGetService:
    CONFIG = HostConfig(host="db.example.com", port=5432, username="user", password="pass")

    @pytest.mark.asyncio
    async def test_creates_service_for_new_database(self):
        with (
            patch.object(server_module, "host_configs", {"db.example.com": self.CONFIG}),
            patch.object(server_module, "db_services", {}),
            patch.object(server_module, "current_access_mode", AccessMode.UNRESTRICTED),
            patch.object(server_module, "query_timeout", None),
        ):
            service = await get_service("mydb")
            assert service.database_url == "postgresql://user:pass@db.example.com:5432/mydb"

    @pytest.mark.asyncio
    async def test_reuses_service_for_same_host_and_database(self):
        with (
            patch.object(server_module, "host_configs", {"db.example.com": self.CONFIG}),
            patch.object(server_module, "db_services", {}),
            patch.object(server_module, "current_access_mode", AccessMode.UNRESTRICTED),
            patch.object(server_module, "query_timeout", None),
        ):
            service1 = await get_service("mydb")
            service2 = await get_service("mydb")
            assert service1 is service2

    @pytest.mark.asyncio
    async def test_different_hosts_get_different_services(self):
        config_b = HostConfig(host="other.example.com", port=5433, username="u2", password="p2")
        configs = {"db.example.com": self.CONFIG, "other.example.com": config_b}
        with (
            patch.object(server_module, "host_configs", configs),
            patch.object(server_module, "db_services", {}),
            patch.object(server_module, "current_access_mode", AccessMode.UNRESTRICTED),
            patch.object(server_module, "query_timeout", None),
        ):
            service_a = await get_service("mydb", "db.example.com")
            service_b = await get_service("mydb", "other.example.com")
            assert service_a is not service_b
            assert "db.example.com" in service_a.database_url
            assert "other.example.com" in service_b.database_url
