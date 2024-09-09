"""
Some functions to parse SVG files
Maybe not worth using, given the plethora of SVG libraries already out there
"""

import numpy as np

numeric = '1234567890.e'

class CubicBezier():
    
    def __init__(self, pts: list[np.ndarray]):
        if len(pts) != 4:
            raise ValueError("Must specify 4 control points")
        self.pts = pts
    
    def evaluate(self, times: float or np.ndarray):
        if isinstance(times, float):
            t = times
            if t == 0:
                return self.pts[0]
            elif t == 1:
                return self.pts[3]
            else:
                return pow(1 - t, 3) * self.pts[0] + 3 * t * pow(1 - t, 2) * self.pts[1] + 3 * pow(t, 2) * (1 - t) * self.pts[2] + pow(t, 3) * self.pts[3]
        else:
            ret = np.zeros((len(times), 2), dtype=np.float64)
            for i,t in enumerate(times):
                if t == 0:
                    ret[i] = self.pts[0]
                elif t == 1:
                    ret[i] = self.pts[3]
                else:
                    ret[i] = pow(1 - t, 3) * self.pts[0] + 3 * t * pow(1 - t, 2) * self.pts[1] + 3 * pow(t, 2) * (1 - t) * self.pts[2] + pow(t, 3) * self.pts[3]
            return ret
        
class QuadraticBezier():
    
    def __init__(self, pts: list[np.ndarray]):
        if len(pts) != 3:
            raise ValueError("Must specify 3 control points")
        self.pts = pts
    
    def evaluate(self, times: float or np.ndarray):
        if isinstance(times, float):
            t = times
            if t == 0:
                return self.pts[0]
            elif t == 1:
                return self.pts[2]
            else:
                return pow(1 - t, 2) * self.pts[0] + 2 * t * (1 - t) * self.pts[1] + pow(t, 2) * self.pts[2]
        else:
            ret = np.zeros((len(times), 2), dtype=np.float64)
            for i,t in enumerate(times):
                if t == 0:
                    ret[i] = self.pts[0]
                elif t == 1:
                    ret[i] = self.pts[2]
                else:
                    ret[i] = pow(1 - t, 2) * self.pts[0] + 2 * t * (1 - t) * self.pts[1] + pow(t, 2) * self.pts[2]
            return ret

class Command():

    def __init__(self, type: str):
        self.type = type
        self.args = []

    def __repr__(self):
        return f"Command {self.type} at {self.args}"

def createCommandStack(cmds: str) -> list[Command]:
    """ creates the command stack for one path object """
    stack = []
    current_cmd = Command(cmds[0])
    current_arg = []
    for i, token in enumerate(cmds[1:]):
        if (token in numeric):
            current_arg.append(token)
        elif (token in ' ,\n'):
            # stash the previous argument
            if current_arg != []:
                current_cmd.args.append(float(''.join(current_arg)))
                current_arg = []
        elif (token in '-'):
            # stash previous arg if necessary
            if current_arg != [] and cmds[i] != 'e':
                current_cmd.args.append(float(''.join(current_arg)))
                current_arg = []
            # start a new argument
            current_arg.append(token)
        else:  
            # stash the previous argument
            if current_arg != []:
                current_cmd.args.append(float(''.join(current_arg)))
                current_arg = []
            # stash the previous command
            stack.append(current_cmd)
            current_cmd = Command(token)
    # stash the last command
    if current_arg != []:
        current_cmd.args.append(float(''.join(current_arg)))
    stack.append(current_cmd)
    current_cmd = Command(token)
    return stack
            
    
def execCommandStack(cmds: list[Command], samples=20):
    """ turns a command stack into a stroke """
    cursor = np.zeros(2)
    pts = []
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for i,cmd in enumerate(cmds):
        ctrl_end = None
        if cmd.type in 'Mm':
            if cmd.type == 'M':
                cursor = np.array([cmd.args[0], cmd.args[1]])
            else:
                cursor += np.array([cmd.args[0], cmd.args[1]])
            continue
        elif cmd.type in 'Cc':
            if cmd.type == 'C':
                bez = CubicBezier(np.vstack([cursor, [cmd.args[0], cmd.args[1]], [cmd.args[2], cmd.args[3]], [cmd.args[4], cmd.args[5]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = np.array([cmd.args[2], cmd.args[3]])
                cursor = np.array([cmd.args[4], cmd.args[5]])
            else:
                bez = CubicBezier(np.vstack([cursor, [cursor[0] + cmd.args[0], cursor[1] + cmd.args[1]], [cursor[0] + cmd.args[2], cursor[1] + cmd.args[3]], 
                                            [cursor[0] + cmd.args[4], cursor[1] + cmd.args[5]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = cursor + np.array([cmd.args[2], cmd.args[3]])
                cursor[0] += cmd.args[4]
                cursor[1] += cmd.args[5]
        elif cmd.type in 'Ss':
            # check the previous command
            if i - 1 > 0 and cmds[i - 1].type in 'CcSs':
                # first control point is a reflection of the last curve's second control point about the cursor
                ctrl1 = cursor + (cursor - new_pts[-2][2])
            else:
                #  first control point is just the cursor
                ctrl1 = cursor
            if cmd.type == 'S':
                bez = CubicBezier(np.vstack([cursor, ctrl1, [cmd.args[0], cmd.args[1]], [cmd.args[2], cmd.args[3]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = np.array([cmd.args[0], cmd.args[1]])
                cursor = np.array([cmd.args[2], cmd.args[3]])
            else:
                bez = CubicBezier(np.vstack([cursor, ctrl1, [cursor[0] + cmd.args[0], cursor[1] + cmd.args[1]], 
                                            [cursor[0] + cmd.args[2], cursor[1] + cmd.args[3]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = cursor + np.array([cmd.args[0], cmd.args[1]])
                cursor[0] += cmd.args[2]
                cursor[1] += cmd.args[3]
        elif cmd.type in 'Qq':
            if cmd.type == 'Q':
                bez = QuadraticBezier(np.vstack([cursor, [cmd.args[0], cmd.args[1]], [cmd.args[2], cmd.args[3]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = np.array([cmd.args[0], cmd.args[1]])
                cursor = np.array([cmd.args[2], cmd.args[3]])
            else:
                bez = QuadraticBezier(np.vstack([cursor, [cursor[0] + cmd.args[0], cursor[1] + cmd.args[1]], [cursor[0] + cmd.args[2], cursor[1] + cmd.args[3]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = cursor + np.array([cmd.args[0], cmd.args[1]])
                cursor[0] += cmd.args[2]
                cursor[1] += cmd.args[3]
        elif cmd.type in 'Tt':
            # check the previous command
            if i - 1 > 0 and cmds[i - 1].type in 'QqTt':
                # control point is a reflection of the last curve's control point about the cursor
                ctrl = cursor + (cursor - new_pts[-2][2])
            else:
                # control point is just the cursor
                ctrl = cursor
            if cmd.type == 'T':
                bez = QuadraticBezier(np.vstack([cursor, ctrl, [cmd.args[0], cmd.args[1]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = ctrl
                cursor = np.array([cmd.args[0], cmd.args[1]])
            else:
                bez = QuadraticBezier(np.vstack([cursor, ctrl, [cursor[0] + cmd.args[0], cursor[1] + cmd.args[1]]]))
                new_pts = bez.evaluate(np.linspace(0, 1, samples))
                ctrl_end = ctrl
                cursor[0] += cmd.args[0]
                cursor[1] += cmd.args[1]
        elif cmd.type in 'Ll':
            if cmd.type == 'L':
                new_pts = np.vstack([cursor, [cmd.args[0], cmd.args[1]]])
                cursor = np.array([cmd.args[0], cmd.args[1]])
            else:
                new_pts = np.vstack([cursor, [cursor[0] + cmd.args[0], cursor[1] + cmd.args[1]]])
                cursor[0] += cmd.args[0]
                cursor[1] += cmd.args[1]
        elif cmd.type in 'Hh':
            if cmd.type == 'H':
                new_pts = np.vstack([cursor, [cmd.args[0], cursor[1]]])
                cursor[0] = cmd.args[0]
            else:
                new_pts = np.vstack([cursor, [cursor[0] + cmd.args[0], cursor[1]]])
                cursor[0] += cmd.args[0]
        elif cmd.type in 'Vv':
            if cmd.type == 'V':
                new_pts = np.vstack([cursor, [cursor[0], cmd.args[0]]])
                cursor[1] = cmd.args[0]
            else:
                new_pts = np.vstack([cursor, [cursor[0], cursor[1] + cmd.args[0]]])
                cursor[1] += cmd.args[0]
        elif cmd.type in 'Zz':
            continue
        else:
            raise ValueError(f"cmd {cmd.type} not handled")
        if len(pts) > 0 and np.all(pts[-1][:2] != new_pts[0]):
            raise ValueError("Stroke is not continuous")
        xmin = min(xmin, min(new_pts[:,0]))
        xmax = max(xmax, max(new_pts[:,0]))
        ymin = min(ymin, min(new_pts[:,1]))
        ymax = max(ymax, max(new_pts[:,1]))
        new_pts = [(pt[0], pt[1], ctrl_end) for pt in new_pts]
        if len(pts) > 0:
            pts += new_pts[1:]
        else:
            pts += new_pts

    return (pts, (xmin, xmax, ymin, ymax))
    
    

