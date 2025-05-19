import type { Dispatch, SetStateAction } from "react";
export interface Project {
  id: string | number;
  title: string;
  status: string;
  startDate: string;
  endDate: string;
  ecMaxContribution: number;
  acronym: string;
  legalBasis: string;
  objective: string;
  frameworkProgramme: string;
  list_euroSciVocTitle: string[];
  list_euroSciVocPath: string[];
}

export interface ProjectDetailsProps {
  project: Project;
  question: string;
  setQuestion: React.Dispatch<React.SetStateAction<string>>;
  askChatbot: () => void;
  chatHistory: ChatMessage[];
  messagesEndRef: React.RefObject<HTMLDivElement>;
}


export interface DashboardProps {
  stats: { [key: string]: { labels: string[]; values: number[] } };
  filters: FilterState;
  setFilters: Dispatch<SetStateAction<FilterState>>;
  availableFilters: AvailableFilters;
  project: Project[];
}

export interface OrganizationLocation {
  name: string;
  country: string;
  latitude: number;
  longitude: number;
}

export interface FilterState {
  [key: string]: string;
  status: string;
  organization: string;
  country: string;
  legalBasis: string;
  minYear: string;
  maxYear: string;
  minFunding: string;
  maxFunding: string;
}


export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AvailableFilters {
  [key: string]: string[];
  statuses: string[];
  organizations: string[];
  countries: string[];
  legalBases: string[];
}

export interface ProjectExplorerProps {
  projects: Project[];
  search: string;
  setSearch: (value: string) => void;
  statusFilter: string;
  setStatusFilter: (value: string) => void;
  page: number;
  setPage: React.Dispatch<React.SetStateAction<number>>;
  setSelectedProject: (project: Project) => void;
  question: string;
  setQuestion: (value: string) => void;
  chatHistory: ChatMessage[];
  askChatbot: () => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
}