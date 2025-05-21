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

  // Debounced stats & filters fetch
  useEffect(() => {
    const qs = new URLSearchParams();
    Object.entries(filters).forEach(([key, val]) => {
      if (val) qs.set(key, val);
    });

    setLoadingStats(true);
    // Fetch stats
    fetch(`/api/stats?${qs.toString()}`)
      .then(res => res.json())
      .then((data: Stats) => setStatsData(data))
      .catch(console.error)
      .finally(() => setLoadingStats(false));

    // Fetch available filters
    fetch(`/api/filters?${qs.toString()}`)
      .then(res => res.json())
      .then((data: AvailableFilters) => setFilters(prev => ({ ...prev, ...{} } as any)))
      .catch(console.error);
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
    'status', 'organization', 'country', 'legalBasis'
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
              : 'legalBases'
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
