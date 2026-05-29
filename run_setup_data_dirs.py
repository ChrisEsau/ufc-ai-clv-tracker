from pipeline.paths import ensure_data_dirs


ensure_data_dirs()

print("Created/verified data directories:")
print("data/master")
print("data/staging")
print("data/audits")
print("data/status")
print("data/backups")