from models import Base
from dependencies.database import engine

Base.metadata.create_all(bind=engine)
print("Database updated successfully")
