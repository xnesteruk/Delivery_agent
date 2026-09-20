class SimulationConfig:
    def __init__(
        self,
        map_width: int = 5,
        map_height: int = 5,
        carrying_capacity: int = 10,
        delivery_allowance_minutes: int = 120,
        seed: int = 42,
    ):
        self.map_width = map_width
        self.map_height = map_height
        self.carrying_capacity = carrying_capacity
        self.delivery_allowance_minutes = delivery_allowance_minutes
        self.seed = seed

        # Fixed rules of our simulation
        self._cell_distance_meters = 500
        self._move_minutes = 6

    @property
    def map_width(self) -> int:
        return self._map_width

    @map_width.setter
    def map_width(self, value: int):
        if value <= 0:
            raise ValueError("Map width must be positive.")
        self._map_width = value

    @property
    def map_height(self) -> int:
        return self._map_height

    @map_height.setter
    def map_height(self, value: int):
        if value <= 0:
            raise ValueError("Map height must be positive.")
        self._map_height = value

    @property
    def carrying_capacity(self) -> int:
        return self._carrying_capacity

    @carrying_capacity.setter
    def carrying_capacity(self, value: int):
        if value <= 0:
            raise ValueError("Carrying capacity must be positive.")
        self._carrying_capacity = value

    @property
    def delivery_allowance_minutes(self) -> int:
        return self._delivery_allowance_minutes

    @delivery_allowance_minutes.setter
    def delivery_allowance_minutes(self, value: int):
        if value <= 0:
            raise ValueError("Delivery allowance must be positive.")
        self._delivery_allowance_minutes = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        if type(value) is not int:
            raise TypeError("Seed must be an integer.")
        self._seed = value

    @property
    def cell_distance_meters(self) -> int:
        return self._cell_distance_meters

    @property
    def move_minutes(self) -> int:
        return self._move_minutes

    @property
    def depot_position(self) -> tuple[int, int]:
        return self.map_width // 2, self.map_height // 2