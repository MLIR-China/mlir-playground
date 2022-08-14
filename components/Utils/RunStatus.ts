export class RunStatus {
  label: string;
  percentage: number;

  constructor(label: string, percentage: number) {
    this.label = label;
    this.percentage = percentage;
  }
}

export type RunStatusListener = (status: RunStatus) => void;
