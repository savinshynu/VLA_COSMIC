import sqlalchemy

from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
from datetime import datetime
engine = CosmicDB_Engine(engine_conf_yaml_filepath="/home/cosmic/conf/cosmicdb_conf.yaml")


with engine.session() as session:
    for result in session.scalars(
        sqlalchemy.select(entities.CosmicDB_Observation)
        .where(
            entities.CosmicDB_Observation.start > datetime.fromisoformat("2023-06-02 00:00:00"),
            entities.CosmicDB_Observation.start < datetime.fromisoformat("2023-06-03 00:00:00"),
        )
        
    ):
        print(result)

"""
with engine.session() as session:
    for result in session.scalars(
        sqlalchemy.select(entities.CosmicDB_ObservationBeam)
        .where(
            entities.CosmicDB_ObservationBeam.start > datetime.fromisoformat("2023-06-03 00:00:00"),
            entities.CosmicDB_ObservationBeam.start < datetime.fromisoformat("2023-06-04 00:00:00"),
            entities.CosmicDB_ObservationBeam.source != 'Incoherent'
        )
        
    ):
        print(result)

"""