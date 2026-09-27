"""Reusable G-code parsing and CSV-to-G-code conversion logic.

Relocated from the non-UI portion of ``gcode_dev.py`` / ``gcode_dev_v2.py``.
Only parsing, coordinate extraction, layer lookup, and CSV-to-G-code
conversion are kept here; the Tkinter editing/UI methods stay with the GUI
entry point under ``scripts/gcode/``.

Public signatures and string/number formatting are preserved exactly as in the
original scripts so generated G-code is byte-identical.
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


class GCodeConverter:
    def __init__(self):
        self.df = None
        self.gcode_lines = []
        self.full_gcode_backup = []  # Simple backup for layer switching
        self.selected_lines = set()
        self.parsed_gcode = []  # Store parsed G-code with coordinates

    def load_csv(self, filepath: str) -> bool:
        """Load CSV file with toolpath data"""
        try:
            self.df = pd.read_csv(filepath)
            required_columns = ['X', 'Y', 'Z', 'Layer', 'Type']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def parse_gcode_line(self, line: str) -> Dict:
        """Parse a G-code line and extract coordinates and commands"""
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('('):
            return {'type': 'comment', 'line': line, 'x': None, 'y': None, 'z': None, 'f': None, 'command': None}

        # Extract coordinates and commands
        result = {'type': 'gcode', 'line': line, 'x': None, 'y': None, 'z': None, 'f': None, 'command': None}

        # Find G-code command
        g_match = re.search(r'G(\d+)', line.upper())
        if g_match:
            result['command'] = f"G{g_match.group(1).zfill(2)}"

        # Extract coordinates
        x_match = re.search(r'X([-+]?\d*\.?\d+)', line.upper())
        y_match = re.search(r'Y([-+]?\d*\.?\d+)', line.upper())
        z_match = re.search(r'Z([-+]?\d*\.?\d+)', line.upper())
        f_match = re.search(r'F([-+]?\d*\.?\d+)', line.upper())

        if x_match:
            result['x'] = float(x_match.group(1))
        if y_match:
            result['y'] = float(y_match.group(1))
        if z_match:
            result['z'] = float(z_match.group(1))
        if f_match:
            result['f'] = float(f_match.group(1))

        return result

    def parse_gcode(self) -> List[Dict]:
        """Parse all G-code lines"""
        self.parsed_gcode = []
        for line in self.gcode_lines:
            self.parsed_gcode.append(self.parse_gcode_line(line))
        return self.parsed_gcode

    def get_gcode_coordinates(self, layer_filter=None) -> Tuple[List, List, List, List]:
        """Extract coordinates from parsed G-code for plotting"""
        x_coords, y_coords, z_coords, commands = [], [], [], []

        # If layer filter is specified, only get coordinates for that layer
        if layer_filter is not None:
            layer_lines = self.get_gcode_lines_for_layer(layer_filter)
            # Parse only the layer-specific lines
            layer_parsed = []
            for line in layer_lines:
                layer_parsed.append(self.parse_gcode_line(line))

            for parsed_line in layer_parsed:
                if parsed_line['type'] == 'gcode' and parsed_line['x'] is not None:
                    x_coords.append(parsed_line['x'])
                    y_coords.append(parsed_line['y'] if parsed_line['y'] is not None else 0)
                    z_coords.append(parsed_line['z'] if parsed_line['z'] is not None else 0)
                    commands.append(parsed_line['command'])
        else:
            # Get all coordinates
            for parsed_line in self.parsed_gcode:
                if parsed_line['type'] == 'gcode' and parsed_line['x'] is not None:
                    x_coords.append(parsed_line['x'])
                    y_coords.append(parsed_line['y'] if parsed_line['y'] is not None else 0)
                    z_coords.append(parsed_line['z'] if parsed_line['z'] is not None else 0)
                    commands.append(parsed_line['command'])

        return x_coords, y_coords, z_coords, commands

    def get_gcode_lines_for_layer(self, layer_num: int) -> List[str]:
        """Extract G-code lines for a specific layer based on layer comments"""
        if not self.gcode_lines:
            return []

        layer_lines = []
        current_layer = None
        in_target_layer = False

        for line in self.gcode_lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            # Check for layer comment (case-insensitive)
            if '; LAYER' in line_upper:
                # Try multiple patterns to extract layer number
                patterns = [
                    r'; LAYER\s*(\d+)',           # "; LAYER 0" or "; LAYER0"
                    r'; LAYER\s*:\s*(\d+)',       # "; LAYER: 0" or "; LAYER:0"
                    r'; LAYER\s*=\s*(\d+)',       # "; LAYER = 0" or "; LAYER=0"
                    r'LAYER\s*(\d+)',             # "LAYER 0" or "LAYER0" anywhere in line
                ]

                layer_found = False
                for pattern in patterns:
                    layer_match = re.search(pattern, line_upper)
                    if layer_match:
                        current_layer = int(layer_match.group(1))
                        in_target_layer = (current_layer == layer_num)
                        layer_found = True

                        if in_target_layer:
                            layer_lines.append(line)  # Include the original layer comment

                            # Handle cases where G-code is attached to layer comment
                            # Look for G-code after the layer comment in the same line
                            gcode_match = re.search(r'(G\d+.*)', line_stripped, re.IGNORECASE)
                            if gcode_match:
                                gcode_part = gcode_match.group(1)
                                if gcode_part.strip():
                                    layer_lines.append(gcode_part)
                        break

                if not layer_found:
                    in_target_layer = False

            elif in_target_layer:
                # Add line if we're in the target layer
                layer_lines.append(line)

        return layer_lines

    def get_available_layers(self) -> List[int]:
        """Get list of available layer numbers from G-code"""
        layers = []

        for line in self.gcode_lines:
            line_stripped = line.strip().upper()  # Convert to uppercase for case-insensitive search

            # Look for "; LAYER" followed by a number (case-insensitive)
            if '; LAYER' in line_stripped:
                # Try multiple patterns to extract layer number
                patterns = [
                    r'; LAYER\s*(\d+)',           # "; LAYER 0" or "; LAYER0"
                    r'; LAYER\s*:\s*(\d+)',       # "; LAYER: 0" or "; LAYER:0"
                    r'; LAYER\s*=\s*(\d+)',       # "; LAYER = 0" or "; LAYER=0"
                    r'LAYER\s*(\d+)',             # "LAYER 0" or "LAYER0" anywhere in line
                ]

                for pattern in patterns:
                    layer_match = re.search(pattern, line_stripped)
                    if layer_match:
                        layer_num = int(layer_match.group(1))
                        if layer_num not in layers:
                            layers.append(layer_num)
                        break  # Found a match, no need to try other patterns

        return sorted(layers)

    def extract_layer_number_from_line(self, line: str) -> Optional[int]:
        """Extract layer number from a line containing layer comment"""
        line_upper = line.strip().upper()

        if '; LAYER' not in line_upper:
            return None

        # Try multiple patterns to extract layer number
        patterns = [
            r'; LAYER\s*(\d+)',           # "; LAYER 0" or "; LAYER0"
            r'; LAYER\s*:\s*(\d+)',       # "; LAYER: 0" or "; LAYER:0"
            r'; LAYER\s*=\s*(\d+)',       # "; LAYER = 0" or "; LAYER=0"
            r'LAYER\s*(\d+)',             # "LAYER 0" or "LAYER0" anywhere in line
        ]

        for pattern in patterns:
            layer_match = re.search(pattern, line_upper)
            if layer_match:
                return int(layer_match.group(1))

        return None

    def convert_to_gcode(self, feedrate_g01: int = 1500, feedrate_g00: int = 3000) -> List[str]:
        """Convert CSV data to G-code"""
        if self.df is None:
            return []

        gcode = []
        gcode.append("; G-code generated from 3D toolpath")
        gcode.append("G21 ; Set units to millimeters")
        gcode.append("G90 ; Absolute positioning")
        gcode.append("M82 ; Set extruder to absolute mode")
        gcode.append("")

        current_layer = -1

        for idx, row in self.df.iterrows():
            x, y, z = row['X'], row['Y'], row['Z']
            layer = row['Layer']
            move_type = row['Type']

            # Layer change comment
            if layer != current_layer:
                gcode.append(f"; Layer {layer}")
                current_layer = layer

            # Generate G-code based on type
            if move_type == 'Deposition':
                gcode.append(f"G01 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feedrate_g01}")
            elif move_type in ['Tool_Move', 'Ramp_Start', 'Ramp_End', 'Fly_Line']:
                gcode.append(f"G00 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feedrate_g00}")
            else:  # Default to travel moves for any other type
                gcode.append(f"G00 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feedrate_g00}")

        gcode.append("")
        gcode.append("M104 S0 ; Turn off hotend")
        gcode.append("M140 S0 ; Turn off bed")
        gcode.append("G28 ; Home all axes")

        self.gcode_lines = gcode
        return gcode

    def get_layer_gcode(self, layer_num: int) -> List[str]:
        """Extract G-code for a specific layer"""
        if self.df is None:
            return []

        layer_data = self.df[self.df['Layer'] == layer_num]
        if layer_data.empty:
            return []

        gcode = []
        gcode.append(f"; G-code for Layer {layer_num}")
        gcode.append("G21 ; Set units to millimeters")
        gcode.append("G90 ; Absolute positioning")
        gcode.append("")

        for idx, row in layer_data.iterrows():
            x, y, z = row['X'], row['Y'], row['Z']
            move_type = row['Type']

            if move_type == 'Deposition':
                gcode.append(f"G01 X{x:.3f} Y{y:.3f} Z{z:.3f} F1500")
            elif move_type in ['Tool_Move', 'Ramp_Start', 'Ramp_End', 'Fly_Line']:
                gcode.append(f"G00 X{x:.3f} Y{y:.3f} Z{z:.3f} F3000")
            else:
                gcode.append(f"G00 X{x:.3f} Y{y:.3f} Z{z:.3f} F3000")

        return gcode

    def regenerate_gcode_from_parsed(self):
        """Regenerate G-code lines from parsed data after modifications"""
        new_gcode_lines = []

        for parsed_line in self.parsed_gcode:
            if parsed_line['type'] == 'comment':
                # Keep comment as-is
                new_gcode_lines.append(parsed_line['line'])
            elif parsed_line['type'] == 'gcode':
                # Reconstruct the G-code line from parsed data
                line_parts = []

                # Add command if present
                if parsed_line['command']:
                    line_parts.append(parsed_line['command'])

                # Add coordinates in standard order
                if parsed_line['x'] is not None:
                    line_parts.append(f"X{parsed_line['x']:.3f}")
                if parsed_line['y'] is not None:
                    line_parts.append(f"Y{parsed_line['y']:.3f}")
                if parsed_line['z'] is not None:
                    line_parts.append(f"Z{parsed_line['z']:.3f}")
                if parsed_line['f'] is not None:
                    line_parts.append(f"F{parsed_line['f']:.3f}")

                new_gcode_lines.append(' '.join(line_parts) if line_parts else parsed_line['line'])
            else:
                # Keep original line for unknown types
                new_gcode_lines.append(parsed_line.get('line', ''))

        self.gcode_lines = new_gcode_lines
