import { useState, useEffect, useRef } from "react";
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
  type ChartData,
  type ChartOptions
} from "chart.js";
import { Bar, Pie, Doughnut, Line } from "react-chartjs-2";
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

type LegendPosition = "top" | "bottom" | "left" | "right" | "chartArea";
interface ChartDataShape { labels: string[]; values: number[]; }
interface Stats { [key: string]: ChartDataShape; }

// define the six charts and their component types
const chartOrder: { key:string; name: string; type: ChartType }[] = [
  { key: "ppy", name: "Projects per Year",          type: "line"     },
  { key: "psd",name: "Project-Size Distribution",  type: "bar"      },
  { key: "frs",name: "Co-funding Ratio by Scheme", type: "bar"      },
  { key: "top10",name: "Top 10 Topics (€ M)",        type: "bar"      },
  { key: "frb",name: "Funding Range Breakdown",    type: "pie"      },
  { key: "ppc",name: "Projects per Country",       type: "doughnut" },
];

const FILTER_LABELS: Record<keyof FilterState, string> = {
  status:       "Status",
  organization: "Organization",
  country:      "Country",
  legalBasis:   "Legal Basis",
  topics:       "Topics",
};

type ChartType = "bar" | "pie" | "doughnut" | "line";

interface DashboardProps {
  stats: Stats;
  filters: FilterState;
  setFilters: React.Dispatch<React.SetStateAction<FilterState>>;
  availableFilters: AvailableFilters;
}

const Dashboard: React.FC<DashboardProps> = ({
  stats: initialStats,
  filters,
  setFilters,
  availableFilters,
}) => {
  const [orgInput, setOrgInput] = useState("");
  const [statsData, setStatsData] = useState<Stats>(initialStats);
  const [loadingStats, setLoadingStats] = useState(false);
  const fetchTimer = useRef<number | null>(null);

  // Debounced stats & filters fetch
  useEffect(() => {
    if (fetchTimer.current) clearTimeout(fetchTimer.current);
    fetchTimer.current = window.setTimeout(() => {
      const qs = new URLSearchParams();
      Object.entries(filters).forEach(([k, v]) => v && qs.set(k, v));

      setLoadingStats(true);
      fetch(`/api/stats?${qs.toString()}`)
        .then(r => r.json())
        .then((data: Stats) => setStatsData(data))
        .catch(console.error)
        .finally(() => setLoadingStats(false));

    }, 300);
    return () => { if (fetchTimer.current) clearTimeout(fetchTimer.current); };
  }, [filters]);


  const updateFilter = (key: keyof FilterState) => 
    (opt: { value: string } | null) => 
      setFilters(prev => ({ ...prev, [key]: opt?.value || "" }));

  const updateSlider = (
    k1: 'minYear' | 'minFunding',
    k2: 'maxYear' | 'maxFunding'
  ) => ([min, max]: number[]) =>
    setFilters(prev => ({ ...prev, [k1]: String(min), [k2]: String(max) }));

  const filterKeys: Array<keyof FilterState> = [
    'status', 'organization', 'country', 'legalBasis','topics'
  ];

  if (loadingStats && !Object.keys(statsData).length) {
    return <Flex justify="center" mt={10}><Spinner size="xl" /></Flex>;
  }

  return (
    <Box>
      {/* Filters */}
      <Box borderWidth="1px" borderRadius="lg" p={4} mb={6} bg="gray.50">
        <Grid templateColumns={{ base: '1fr', sm: 'repeat(2,1fr)', md: 'repeat(4,1fr)', lg: 'repeat(6,1fr)' }} gap={4}>
          {filterKeys.map(key => {
            const opts = availableFilters[
              key === 'status' ? 'statuses'
              : key === 'organization' ? 'organizations'
              : key === 'country' ? 'countries'
              : key === "legalBasis" ? "legalBases"
              : 'topics'
            ] || [];
            const isOrg = key === 'organization';
            return (
              <GridItem key={key} colSpan={1}>
                <Text fontSize="sm" mb={1} fontWeight="medium">{FILTER_LABELS[key]}</Text>
                <Select
                  options={opts.map(v => ({ label: v, value: v }))}
                  placeholder={FILTER_LABELS[key]}
                  onChange={updateFilter(key)}
                  isClearable
                  isSearchable
                  {...(isOrg && { menuIsOpen: orgInput.length>0, onInputChange: setOrgInput })}
                />
              </GridItem>
            );
          })}
          {/* Year Range */}
          <GridItem colSpan={{ base: 1, md: 2 }}>
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
                  <RangeSliderFilledTrack />
                </RangeSliderTrack>
                <RangeSliderThumb index={0} boxSize={4}/>
                <RangeSliderThumb index={1} boxSize={4}/>
              </RangeSlider>
            </Box>
          </GridItem>

          {/* Funding Range */}
          <GridItem colSpan={{ base: 1, md: 2 }}>
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
                max={1e7}
                step={1e5}
                defaultValue={[+filters.minFunding, +filters.maxFunding]}
                onChange={updateSlider("minFunding","maxFunding")}
                size="md"
              >
                <RangeSliderTrack>
                  <RangeSliderFilledTrack />
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
        {chartOrder.map(({key,name, type }) => {
          const raw = statsData[key]!;
          if (!raw) return null;
          // ---- properly typed Chart.js data & options ----
          if (type === "bar") {
            const data: ChartData<"bar", number[], string> = {
              labels: raw.labels,
              datasets: [
                {
                  label: name,
                  data: raw.values,
                  backgroundColor: "#003399",
                  borderColor: "#FFCC00",
                  borderWidth: 1,
                },
              ],
            };
            const options: ChartOptions<"bar"> = {
              responsive: true,
              plugins: {
                legend: {
                  position: "top" as LegendPosition,
                },
                title: {
                  display: true,
                  text: name,
                },
              },
            };
            return (
              <Box key={name} bg="white" borderRadius="md" p={4}>
                <Bar data={data} options={options} />
              </Box>
            );
          }
          if (type === "line") {
            const data: ChartData<"line", number[], string> = {
              labels: raw.labels,
              datasets: [
                {
                  label: name,
                  data: raw.values,
                  backgroundColor: "#003399",
                  borderColor: "#FFCC00",
                  borderWidth: 1,
                },
              ],
            };
            const options: ChartOptions<"line"> = {
              responsive: true,
              plugins: {
                legend: {
                  position: "top" as LegendPosition,
                },
                title: {
                  display: true,
                  text: name,
                },
              },
            };
            return (
              <Box key={name} bg="white" borderRadius="md" p={4}>
                <Line data={data} options={options} />
              </Box>
            );
          }
          if (type === "pie") {
            const data: ChartData<"pie", number[], string> = {
              labels: raw.labels,
              datasets: [
                {
                  label: name,
                  data: raw.values,
                  backgroundColor: "#003399",
                  borderColor: "#FFCC00",
                  borderWidth: 1,
                },
              ],
            };
            const options: ChartOptions<"pie"> = {
              responsive: true,
              plugins: {
                legend: {
                  position: "top" as LegendPosition,
                },
                title: {
                  display: true,
                  text: name,
                },
              },
            };
            return (
              <Box key={name} bg="white" borderRadius="md" p={4}>
                <Pie data={data} options={options} />
              </Box>
            );
          }
          if (type === "doughnut") {
            const data: ChartData<"doughnut", number[], string> = {
              labels: raw.labels,
              datasets: [
                {
                  label: name,
                  data: raw.values,
                  backgroundColor: "#003399",
                  borderColor: "#FFCC00",
                  borderWidth: 1,
                },
              ],
            };
            const options: ChartOptions<"doughnut"> = {
              responsive: true,
              plugins: {
                legend: {
                  position: "top" as LegendPosition,
                },
                title: {
                  display: true,
                  text: name,
                },
              },
            };
            return (
              <Box key={name} bg="white" borderRadius="md" p={4}>
                <Doughnut data={data} options={options} />
              </Box>
            );
          }
          return null;
        })}
      </SimpleGrid>
    </Box>
  );
};

export default Dashboard;
