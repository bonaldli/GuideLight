#!/usr/bin/env python3
# encoding: utf-8

import copy
import os
import subprocess
import time
import math
import xml.etree.ElementTree
import numpy as np
# import libsumo
import traci
import sumolib
from sumolib import checkBinary
from sumo_files.env.intersection import Intersection


class TSCSimulator:
    def __init__(self, config, port=None, not_default=True):
        self.not_default = not_default
        self.port = port
        self.name = config.get("name")
        self.seed = config.get('seed', 777)
        self.agent = config.get('agent')
        self.is_libsumo = config.get('is_libsumo', True)
        self._yellow_duration = config.get("yellow_duration")
        self.iter_get_flow = config.get("iter_get_flow")
        self.is_record = config.get("is_record")
        self.config = config
        self.output_path = config.get("output_path")
        self.reward_type = config.get("reward_type")
        self.is_neighbor_reward = config.get("is_neighbor_reward", False)
        self.step_num = config.get("step_num", 1)
        self.p = config.get("p", 1) #for trip.xml output dir
        self._current_time = config.get("episode_start_time", 21600)
        self._init_sim(config.get("sumocfg_file"), self.seed,
                       config.get("episode_length_time"), config.get("gui"))
        self.all_tls = list(self.sim.trafficlight.getIDList())
        self.infastructure = self._infastructure_extraction1(config.get("sumocfg_file"))
        # delete invalid tl part 1
        rm_num = 0
        ora_num = len(self.all_tls)
        for i in self.infastructure:
            if self.infastructure[i]['entering_lanes_pos'] == []:
                rm_num += 1
                self.all_tls.remove(i)
                print("no infastrurct: {}".format(i))
        for rtl in ['1673431902', '8996', '9153', '9531', '9884', '7223736528']:
            if rtl in self.all_tls:
                rm_num += 1
                self.all_tls.remove(rtl)
        rm_tl = []
        self.vehicle_info = {}
        self._crosses = {}
        self.state_key = config['state_key']
        self.all_reward = {}
        self.tl_phase_index = []
        for tl in self.all_tls:
            self._crosses[tl] = Intersection(tl, self.infastructure[tl],
                                             self, self.state_key, self.not_default)
            tl_ava = self._crosses[tl].get_tl_ava()
            if not tl_ava:
                rm_tl.append(tl)
                rm_num += 1
                print("Not ava: {}".format(tl))
        # delete invalid tl part 2
        for tl in rm_tl:
            del self._crosses[tl]
            self.all_tls.remove(tl)
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl].update_timestep()
            self.tl_phase_index.append(self._crosses[tl].get_phase_index())

        print("Remove {} tl, percent: {}".format(rm_num, rm_num / ora_num))

        self.infastructure = self._infastructure_extraction2(config.get("sumocfg_file"),
                                                             self.infastructure,
                                                             config.get("is_dis", False))


        self.terminate()

    def _init_sim(self, sumocfg_file, seed, episode_length_time, gui=False):
        # self.episode_length_time = episode_length_time
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        if sumocfg_file.split("/")[-1] in ['grid4x4.sumocfg', 'arterial4x4.sumocfg', 'base.sumocfg']:
            route_name = sumocfg_file.split("/")[-1][:-8]
            net = "/".join(sumocfg_file.split("/")[:-1] + ['base_v2.net.xml'])
            route = "/".join(sumocfg_file.split("/")[:-2] + ["sumo_fenglin_base_sub1", route_name])
            command = [
                checkBinary(app), '-n', net, '-r', route + '_' + str(self.step_num) + '.rou.xml']
            command += ['-a', "/".join(sumocfg_file.split("/")[:-1] + ['e1.add.xml']) + ", " +
                        "/".join(sumocfg_file.split("/")[:-1] + ['e2.add.xml'])]
        else:
            command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--random']
        # command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport',
                        '600']  # long teleport for safety
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            if not os.path.exists(self.output_path):
                try:
                    os.mkdir(self.output_path)
                except:
                    pass
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_%s_trip.xml' % (
                            self.name, sumocfg_file.split("/")[-1], self.p)),
                        '--tripinfo-output.write-unfinished']
        if self.is_libsumo:
            libsumo.start(command)
            self.sim = libsumo
        else:
            command += ['--remote-port', str(self.port)]
            subprocess.Popen(command)
            time.sleep(2)
            self.sim = traci.connect(port=self.port, numRetries=1000)
        self.step_num += 1
        self.p += 1
        self.episode_length_time = episode_length_time + self._current_time
        self.sim.simulationStep(self._current_time)

    def terminate(self):
        self.sim.close()

    def _do_action(self, action):
        for tl, a in action.items():
            self._crosses[tl].set_phase(a)

    def _get_reward(self):

        list_reward = {tl: self._crosses[tl].get_reward(self.reward_type) for tl in self.all_tls}

        return list_reward

    def step(self, action):
        self._do_action(action)
        for tl in self.all_tls:
            self._crosses[tl].update_timestep()
        done = False
        obs = self._get_state()
        reward_ = self._get_reward()
        # multi-agent reward sum with neighbor
        if self.is_neighbor_reward:
            reward = {}
            for tl in self.all_tls:
                reward[tl] = {}
                for k in reward_[tl]:
                    reward[tl][k] = reward_[tl][k]
                    count = 0
                    tmp = 0
                    for nei in self._crosses[tl].nearset_inter[0][0]:
                        if nei != -1:
                            count += 1
                            tmp += reward_[self.all_tls[nei]][k]
                    if count > 0:
                        reward[tl][k] += tmp / count
        else:
            reward = reward_
        for tl, v in reward.items():
            for k, r in v.items():
                self.all_reward[tl][k] += r
        if self._current_time >= self.episode_length_time:
            done = True
            self.terminate()
        return obs, reward, done, self.all_reward

    def default_step(self):
        done = False
        obs = self._get_state()
        reward = self._get_reward()
        for tl, v in reward.items():
            for k, r in v.items():
                self.all_reward[tl][k] += r
        if self._current_time >= self.episode_length_time:
            done = True
            self.terminate()

        return obs, reward, done, self.all_reward

    def reset(self):
        """have to terminate previous sim before calling reset"""
        self._current_time = self.config.get("episode_start_time", 21600)
        self._init_sim(self.config.get("sumocfg_file"), self.seed,
                       self.config.get("episode_length_time"), self.config.get("gui"))
        self.step_num = self.step_num % 1400 + 1
        # self.infastructure = self._infastructure_extraction1(self.cfg.get("sumocfg_file"))
        self.vehicle_info.clear()
        self.all_reward = {}
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl] = Intersection(tl, self.infastructure[tl], self, self.state_key, self.not_default)
            self._crosses[tl].update_timestep()
        return self._get_state()

    def reset_default(self):
        self._current_time = self.config.get("episode_start_time", 21600)
        self._init_sim(self.config.get("sumocfg_file"), self.seed,
                       self.config.get("episode_length_time"), self.config.get("gui"))
        self.infastructure = self._infastructure_extraction1(self.config.get("sumocfg_file"))
        self.vehicle_info.clear()
        self.all_reward = {}
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl] = Intersection(tl, self.infastructure[tl], self, self.state_key,
                                             self.not_default)
            self._crosses[tl].update_timestep()
        return self._get_state()

    def _get_state(self):
        total_interval = 60 * 15
        assert total_interval % self.iter_get_flow == 0
        flows = {}
        queue_length = {}
        spillback = {}
        for i in range(int(total_interval/self.iter_get_flow)):
            self._current_time += self.iter_get_flow
            self.sim.simulationStep(self._current_time)
            for tid in self.all_tls:
                if self._current_time % 120 == self.iter_get_flow:
                    if tid not in spillback:
                        spillback[tid] = []
                    spillback[tid].append(self._crosses[tid].get_spillback())
                self._crosses[tid].update_timestep()
                if tid not in flows:
                    flows[tid] = []
                    queue_length[tid] = []
                flows[tid].append(self._crosses[tid].get_lane_traffic_volumn())
                queue_length[tid].append(self._crosses[tid].get_lane_queue_len())
        for tid in self.all_tls:
            self._crosses[tid].set_interval_flow(flows[tid])
            self._crosses[tid].set_interval_spillback(spillback[tid])
            self._crosses[tid].set_interval_queue_length(queue_length[tid])

        state = {}
        for tid in self.all_tls:
            state[tid] = self._crosses[tid].get_state()

        return state

    def _infastructure_extraction1(self, sumocfg_file):
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        network_file = os.path.join(os.path.split(sumocfg_file)[0], network_file_name)
        net = xml.etree.ElementTree.parse(network_file).getroot()

        traffic_light_node_dict = {}
        for tl in net.findall("tlLogic"):
            if tl.attrib['id'] not in traffic_light_node_dict.keys():
                node_id = tl.attrib['id']
                traffic_light_node_dict[node_id] = {'leaving_lanes': [], 'entering_lanes': [],
                                                    'leaving_lanes_pos': [], 'entering_lanes_pos': [],
                                                    # "total_inter_num": None,
                                                    'adjacency_row': None}
                traffic_light_node_dict[node_id]["phases"] = [child.attrib["state"] for child in tl]

        # for index, item in enumerate(traffic_light_node_dict):
        #     traffic_light_node_dict[item]['total_inter_num'] = total_inter_num

        for edge in net.findall("edge"):
            if not edge.attrib['id'].startswith(":"):
                if edge.attrib['from'] in traffic_light_node_dict.keys():
                    for child in edge:
                        if "id" in child.keys() and child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['from']]['leaving_lanes'].append(
                                child.attrib['id'])
                            traffic_light_node_dict[edge.attrib['from']]['leaving_lanes_pos'].append(child.attrib['shape'])
                if edge.attrib['to'] in traffic_light_node_dict.keys():
                    for child in edge:
                        if "id" in child.keys() and child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['to']]['entering_lanes'].append(child.attrib['id'])
                            traffic_light_node_dict[edge.attrib['to']]['entering_lanes_pos'].append(child.attrib['shape'])

        for junction in net.findall("junction"):
            if junction.attrib['id'] in traffic_light_node_dict.keys():
                traffic_light_node_dict[junction.attrib['id']]['location'] = \
                    {'x': float(junction.attrib['x']), 'y': float(junction.attrib['y'])}

        return traffic_light_node_dict

    def bfs_find_neighbor(self, queue, edge_dict, net_lib, tl):
        seen = set()
        seen.add(tl)
        parents = {tl: None}
        while len(queue) > 0:
            if edge_dict[queue[0]].attrib['from'] != tl and edge_dict[queue[0]].attrib['from'] not in self.all_tls:
                next_node = edge_dict[queue[0]].attrib['from']
                if next_node not in seen:
                    seen.add(next_node)
                    next_tmp_edges = net_lib.getNode(next_node).getIncoming()
                    next_edges = {}
                    for nte in next_tmp_edges:
                        if edge_dict[nte._id].attrib['from'] == edge_dict[queue[0]].attrib['to']:
                            continue
                        # net_lib.getShortestPath(SC, CE)
                        c1 = net_lib.getNode(edge_dict[nte._id].attrib['from']).getCoord()
                        c2 = net_lib.getNode(next_node).getCoord()
                        c1 = {"x": c1[0], "y": c1[1]}
                        c2 = {"x": c2[0], "y": c2[1]}
                        next_edges[nte._id] = self._cal_distance(c1, c2)
                    if len(next_edges) > 0:
                        next_edges = sorted(next_edges.items(), key=lambda item: item[1])
                        next_edges = [ne[0] for ne in next_edges]
                    queue.extend(next_edges)
                    parents[next_node] = edge_dict[queue[0]].attrib['to']
                del queue[0]
            else:
                parents[edge_dict[queue[0]].attrib['from']] = edge_dict[queue[0]].attrib['to']
                return True, parents, edge_dict[queue[0]].attrib['from']
        return False, None, None
            # length += self._cal_distance(
            #     traffic_light_node_dict[
            #         edge_dict[entering_sequence[es]].attrib['from']]['location'],
            #     traffic_light_node_dict[
            #         edge_dict[entering_sequence[es]].attrib['to']]['location']
            # )

    def _infastructure_extraction2(self, sumocfg_file, traffic_light_node_dict, dis=False):
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        network_file = os.path.join(os.path.split(sumocfg_file)[0], network_file_name)
        net = xml.etree.ElementTree.parse(network_file).getroot()
        net_lib = sumolib.net.readNet(network_file)

        # all_tls is deleted
        all_traffic_light = self.all_tls
        total_inter_num = len(self.all_tls)
        if dis:
            top_k = 5
            for i in range(total_inter_num):
                if 'location' not in traffic_light_node_dict[all_traffic_light[i]]:
                    continue
                location_1 = traffic_light_node_dict[all_traffic_light[i]]['location']
                row = np.array([0] * total_inter_num)
                for j in range(total_inter_num):
                    if 'location' not in traffic_light_node_dict[all_traffic_light[j]]:
                        row[j] = 1e8
                        continue
                    location_2 = traffic_light_node_dict[all_traffic_light[j]]['location']
                    dist = self._cal_distance(location_1, location_2)
                    row[j] = dist if dis < 700 else 1e8
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = list(range(total_inter_num))

                adjacency_row_unsorted.remove(i)
                for j in range(len(adjacency_row_unsorted)):
                    if row[adjacency_row_unsorted[j]] > 700:
                        adjacency_row_unsorted[j] = -1
                # adjacency_row_unsorted = [j for j in adjacency_row_unsorted]
                traffic_light_node_dict[all_traffic_light[i]]['adjacency_row'] = \
                    [[adjacency_row_unsorted, row[adjacency_row_unsorted]], []]
        else:
            # adjacency_row is [entering NWSE neighbor, outgoing NWSE neighbor]
            #  NWSE neighbor contain: 1. neighbor tl name 2. edge distance

            edge_dict = {}
            for edge in net.findall("edge"):
                edge_dict[edge.attrib['id']] = edge

            junction_dict = {}
            for jun in net.findall("junction"):
                junction_dict[jun.attrib["id"]] = jun

            for i in self.all_tls:
                # if i == '89173763':
                # if i == 'J36':
                #     print(1)
                entering_sequence_NWSE = self._crosses[i].entering_sequence_NWSE
                entering_sequence = self._crosses[i].entering_sequence
                outgoing_sequence_NWSE = self._crosses[i].outgoing_sequence_NWSE
                outgoing_sequence = self._crosses[i].outgoing_sequence
                adjacency_row_entering = []
                adjacency_distance_entering = []
                adjacency_row_outgoing = []
                adjacency_distance_outgoing = []
                for es in entering_sequence_NWSE:
                    if es != -1:
                        queue = [entering_sequence[es]]
                        flag, parents, last_node = self.bfs_find_neighbor(queue, edge_dict, net_lib, i)
                        # while edge_dict[queue[0]].attrib['from'] not in all_traffic_light:
                        #     next_node = edge_dict[entering_sequence[es]].attrib['from']
                        #     next_tmp_edges = net_lib.getNode(next_node).getIncoming()
                        #     for index, nte in enumerate(next_tmp_edges):
                        #         if edge_dict[nte].attrib['from'] == i:
                        #             rm_index = index
                        #             break
                        #     del next_tmp_edges[rm_index]
                        #     queue.extend(next_tmp_edges)
                        if flag:
                            length = 0
                            last_node_ = last_node
                            while parents[last_node] != None:
                                coor1 = net_lib.getNode(last_node).getCoord()
                                coor2 = net_lib.getNode(parents[last_node]).getCoord()
                                c1 = {"x": coor1[0], "y": coor1[1]}
                                c2 = {"x": coor2[0], "y": coor2[1]}
                                length += self._cal_distance(c1, c2)
                                if length > 700:
                                    break
                                last_node = parents[last_node]
                            if length > 700:
                                no_neigh = True
                            else:
                                no_neigh = False
                        else:
                            no_neigh = True
                    else:
                        no_neigh = True
                    if no_neigh:
                        adjacency_row_entering.append(-1)
                        adjacency_distance_entering.append(1e8)
                    else:
                        adjacency_row_entering.append(
                            self.all_tls.index(last_node_))
                        adjacency_distance_entering.append(length / 100)
                for os_ in outgoing_sequence_NWSE:
                    if os_ != -1 and edge_dict[outgoing_sequence[os_]].attrib['to'] in all_traffic_light:
                        adjacency_row_outgoing.append(
                            self.all_tls.index(edge_dict[outgoing_sequence[os_]].attrib['to']))
                        adjacency_distance_outgoing.append(self._cal_distance(
                            traffic_light_node_dict[
                                edge_dict[outgoing_sequence[os_]].attrib['from']]['location'],
                            traffic_light_node_dict[
                                edge_dict[outgoing_sequence[os_]].attrib['to']]['location']
                        )/100)
                    else:
                        adjacency_row_outgoing.append(-1)
                        adjacency_distance_outgoing.append(1e8)
                traffic_light_node_dict[i]['adjacency_row'] = \
                    [(adjacency_row_entering, adjacency_distance_entering),
                     (adjacency_row_outgoing, adjacency_distance_outgoing)]

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def _coordinate_sequence(list_coord_str):
        import re
        list_coordinate = [re.split(r'[ ,]', lane_str) for lane_str in list_coord_str]
        # x coordinate
        x_all = np.concatenate(list_coordinate).astype('float64')
        west = np.int(np.argmin(x_all)/2)

        y_all = np.array(list_coordinate, dtype=float)[:, [1, 3]]

        south = np.int(np.argmin(y_all)/2)

        east = np.int(np.argmax(x_all)/2)
        north = np.int(np.argmax(y_all)/2)

        list_coord_sort = [west, north, east, south]
        return list_coord_sort

    @staticmethod
    def _sort_lane_id_by_sequence(ids,sequence=[2, 3, 0, 1]):
        result = []
        for i in sequence:
            result.extend(ids[i*3: i*3+3])
        return result

    @staticmethod
    def get_actual_lane_id(lane_id_list):
        actual_lane_id_list = []
        for lane_id in lane_id_list:
            if not lane_id.startswith(":"):
                actual_lane_id_list.append(lane_id)
        return actual_lane_id_list


if __name__ == '__main__':
    config = {
        "name": "test",
        "agent": "",
        # "sumocfg_file": "sumo_files/scenarios/nanshan/osm.sumocfg",
        "sumocfg_file": "sumo_files/scenarios/sumo_fenglin_base_sub3/base.sumocfg",
        #"sumocfg_file": "sumo_files/scenari os/resco_envs/arterial4x4/arterial4x4.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg",
        "action_type": "select_phase",
        "is_dis": True,
        "is_libsumo": False,
        "iter_get_flow": 15,
        "gui": False,
        "yellow_duration": 3,
        "episode_length_time": 86400,
        "is_record": False,
        'reward_type': ['flow', 'green_utilization', 'green_balance','queue_len'],
        'state_key': ['flow', 'spillback', 'road_capacity', 'current_phase', 'duration',
                      'green_utilization', 'green_balance']
    }
    env = TSCSimulator(config, 12345)

    # now = 0
    # start = time.time()
    # while now < 3600:
    #     env.sim.simulationStep()
    #     now+=1
    # end = time.time()
    # print("one step: {}".format(end-start))
    #
    # now = 0
    # start = time.time()
    # while now < 3600:
    #     env.sim.simulationStep(15)
    #     now+=15
    # end = time.time()
    # print("15 step: {}".format(end - start))

    all_tl = env.all_tls
    for i in range(5):
        env.reset()
        done = False
        while not done:
            done, reward = env.default_step()
        print(reward)
        env.terminate()
        r = {}
        for tl, rs in reward.items():
            for k, v in rs.items():
                if k not in r:
                    r[k] = []
                r[k].append(v)
        total_reward = 0
        for k, v in r.items():
            total_reward += np.mean(v)
            print('Step{}, reward_{}: {}'.format(i, k, np.mean(v)))
        print('Step{}, reward/all : {}'.format(i, total_reward))
    # for i in range(3):
    #     tl_action_select = {}
    #     for tl in all_tl:
    #         a = np.random.choice(12)
    #         while a / 3 in env._crosses[tl].unava_index:
    #             a = np.random.choice(env._crosses[tl].green_phases)
    #         tl_action_select[tl] = a
    #     next_obs, reward, done, _ = env.step(tl_action_select)
    # env.terminate()

