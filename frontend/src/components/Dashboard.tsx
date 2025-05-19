import { useState, useEffect, useRef } from "react";//React, 
import {
  Box,
  Grid,
  GridItem,
  Text,
  Flex,
  Spinner,
  SimpleGrid,
  RangeSlider,
  RangeSliderTrack,
  RangeSliderFilledTrack,
  RangeSliderThumb,
} from "@chakra-ui/react";
import Select from "react-select";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  LineElement,
  PointElement,
  RadialLinearScale,
} from "chart.js";
import { Bar, Pie, Doughnut, Line, Radar, PolarArea } from "react-chartjs-2";
import type { FilterState, AvailableFilters } from "../hooks/types";

// register chart components
ChartJS.register(
  BarElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  LineElement,
  PointElement,
  RadialLinearScale
);

interface ChartData { labels: string[]; values: number[]; }
interface Stats { [key: string]: ChartData; }
const FILTER_LABELS: Record<keyof FilterState, string> = {
  status:       "Status",
  organization: "Organization",
  country:      "Country",
  legalBasis:   "Legal Basis",
};
interface DashboardProps {
  stats: Stats;
  filters: FilterState;
  setFilters: React.Dispatch<React.SetStateAction<FilterState>>;
  availableFilters: AvailableFilters;
}

const chartTypes = ["bar","pie","doughnut","line","radar","polarArea"] as const;
type ChartType = typeof chartTypes[number];

const Dashboard: React.FC<DashboardProps> = ({
  stats: initialStats,
  filters,
  setFilters,
  availableFilters,
}) => {
  const [orgInput, setOrgInput] = useState("");
  const [statsData, setStatsData] = useState<Stats>(initialStats);
  const [loadingStats, setLoadingStats] = useState(false);

  // ref to hold our debounce timer
  const fetchTimer = useRef<number | null>(null);

  useEffect(() => {
    // clear any pending fetch
    if (fetchTimer.current) {
      clearTimeout(fetchTimer.current);
    }

    // schedule new fetch 300ms after last filter change
    fetchTimer.current = window.setTimeout(() => {
      const qs = new URLSearchParams();
      if (filters.status)       qs.set("status", filters.status);
      if (filters.organization) qs.set("organization", filters.organization);
      if (filters.country)      qs.set("country", filters.country);
      if (filters.legalBasis)   qs.set("legalBasis", filters.legalBasis);
      qs.set("minYear",    filters.minYear);
      qs.set("maxYear",    filters.maxYear);
      qs.set("minFunding", filters.minFunding);
      qs.set("maxFunding", filters.maxFunding);

      setLoadingStats(true);
      fetch(`/api/stats?${qs.toString()}`)
        .then(res => res.json())
        .then((data: Stats) => setStatsData(data))
        .catch(console.error)
        .finally(() => setLoadingStats(false));
    }, 300);

    // cleanup on unmount or next effect-run
    return () => {
      if (fetchTimer.current) {
        clearTimeout(fetchTimer.current);
      }
    };
  }, [
    filters.status,
    filters.organization,
    filters.country,
    filters.legalBasis,
    filters.minYear,
    filters.maxYear,
    filters.minFunding,
    filters.maxFunding,
  ]);

  const updateFilter = (key: keyof FilterState) => (opt: { value: string } | null) =>
    setFilters(prev => ({ ...prev, [key]: opt?.value || "" }));

  const updateSlider = (
    k1: "minYear" | "minFunding",
    k2: "maxYear" | "maxFunding"
  ) => ([min, max]: number[]) =>
    setFilters(prev => ({
      ...prev,
      [k1]: String(min),
      [k2]: String(max),
    }));

  const filterKeys: Array<keyof FilterState> = [
    "status",
    "organization",
    "country",
    "legalBasis",
  ];

  // initial blank-state spinner
  if (loadingStats && !Object.keys(statsData).length) {
    return (
      <Flex justify="center" mt={10}>
        <Spinner size="xl" />
      </Flex>
    );
  }

  return (
    <Box>
      {/* Filters */}
      <Box borderWidth="1px" borderRadius="lg" p={4} mb={6} bg="gray.50">
        <Grid
          templateColumns={{
            base: "repeat(1,1fr)",
            sm: "repeat(2,1fr)",
            md: "repeat(3,1fr)",
            lg: "repeat(6,1fr)",
          }}
          gap={4}
          columnGap={6}
        >
          {filterKeys.map((key) => {
            const isOrg = key === "organization";
            const opts =
              availableFilters[
                key === "status"
                  ? "statuses"
                  : key === "organization"
                  ? "organizations"
                  : key === "country"
                  ? "countries"
                  : "legalBases"
              ] || [];

            return (
              <GridItem key={key} colSpan={1}>
                <Text fontSize="sm" mb={1} fontWeight="medium">
                  {FILTER_LABELS[key]}
                </Text>
                <Select
                  options={opts.map(v => ({ label: v, value: v }))}
                  placeholder={`Select ${key}`}
                  onChange={updateFilter(key)}
                  isClearable
                  isSearchable
                  openMenuOnClick
                  openMenuOnFocus
                  {...(isOrg && {
                    openMenuOnClick: false,
                    openMenuOnFocus: false,
                    menuIsOpen: orgInput.length > 0,
                    onInputChange: (str: string) => setOrgInput(str),
                  })}
                />
              </GridItem>
            );
          })}

          {/* Year Range */}
          <GridItem colSpan={{ base: 1, md: 3 }}>
            <Box mb={6}>
              <Flex justify="space-between" mb={1}>
                <Text fontSize="sm" fontWeight="medium">Year Range</Text>
                <Text fontSize="xs" color="gray.600">
                  {filters.minYear} – {filters.maxYear}
                </Text>
              </Flex>
              <RangeSlider
                aria-label={["Min Year","Max Year"]}
                min={2000}
                max={2025}
                step={1}
                defaultValue={[+filters.minYear, +filters.maxYear]}
                onChange={updateSlider("minYear","maxYear")}
                size="md"
              >
                <RangeSliderTrack>
                  <RangeSliderFilledTrack bg="brand.blue" />
                </RangeSliderTrack>
                <RangeSliderThumb index={0} boxSize={4}/>
                <RangeSliderThumb index={1} boxSize={4}/>
              </RangeSlider>
            </Box>
          </GridItem>

          {/* Funding Range */}
          <GridItem colSpan={{ base: 1, md: 3 }}>
            <Box>
              <Flex justify="space-between" mb={1}>
                <Text fontSize="sm" fontWeight="medium">Funding (€)</Text>
                <Text fontSize="xs" color="gray.600">
                  €{Number(filters.minFunding).toLocaleString()} – €{Number(filters.maxFunding).toLocaleString()}
                </Text>
              </Flex>
              <RangeSlider
                aria-label={["Min Funding","Max Funding"]}
                min={0}
                max={10_000_000}
                step={100_000}
                defaultValue={[+filters.minFunding, +filters.maxFunding]}
                onChange={updateSlider("minFunding","maxFunding")}
                size="md"
              >
                <RangeSliderTrack>
                  <RangeSliderFilledTrack bg="brand.blue" />
                </RangeSliderTrack>
                <RangeSliderThumb index={0} boxSize={4}/>
                <RangeSliderThumb index={1} boxSize={4}/>
              </RangeSlider>
            </Box>
          </GridItem>
        </Grid>
      </Box>

      {/* Charts */}
      {loadingStats && (
        <Flex justify="center" mb={6}>
          <Spinner />
        </Flex>
      )}
      <SimpleGrid columns={{ base:1, md:2, lg:3 }} spacing={6}>
        {Object.entries(statsData).map(([label, data], idx) => {
          const type = chartTypes[idx % chartTypes.length] as ChartType;
          const chartProps = {
            data: { labels: data.labels, datasets: [{ label, data: data.values, backgroundColor: "#003399", borderColor: "#FFCC00", borderWidth: 1 }] },
            options: { responsive: true, plugins: { legend: { position: "top" as const }, title: { display: true, text: label } } }
          };

          return (
            <Box key={label} bg="white" borderRadius="md" p={4}>
              {type === "bar"       && <Bar {...chartProps} />}
              {type === "pie"       && <Pie {...chartProps} />}
              {type === "doughnut"  && <Doughnut {...chartProps} />}
              {type === "line"      && <Line {...chartProps} />}
              {type === "radar"     && <Radar {...chartProps} />}
              {type === "polarArea" && <PolarArea {...chartProps} />}
            </Box>
          );
        })}
      </SimpleGrid>
    </Box>
  );
};

export default Dashboard;
