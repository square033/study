#include <iostream>
#include <vector>
using namespace std;

#define INF 1e9

int n, m, start; // 노드의 개수 n, 간선의 개수 m, 시작 노드 번호 start

vector<pair<int, int> > node[100001]; // 연결된 노드 정보 담는 배열

bool visited[100001]; // 방문 유무 배열
int dis[100001]; // 최단 거리 배열

void dijkstra(){
    // 처음 방문한 노드 방문 처리 및 weight 0 지정
    dis[start] = 0;
    visited[start] = true;

    // start와 직접 연결된 노드들의 거리 초기화
    for (auto &edge : node[start]) {
        int next = edge.first;
        int weight = edge.second;
        if (dis[start] + weight < dis[next]) {
            dis[next] = dis[start] + weight;
        }
    }

    int visitedCount = 1; // start 노드는 이미 방문 처리됨

    while (visitedCount < n) {
        // 1. 방문하지 않은 노드 중 dist가 가장 작은 노드 선택 (표의 "min")
        int cur = -1;
        int minDist = INF;
        for (int i = 1; i <= n; i++) {
            if (!visited[i] && dis[i] < minDist) {
                minDist = dis[i];
                cur = i;
            }
        }

        if (cur == -1) break; // 남은 노드에 도달 불가능

        visited[cur] = true; // 표의 "X" 처리
        visitedCount++;

        // 2. cur과 연결된 인접 노드들의 거리 갱신 (표의 "+" 계산, min으로 갱신)
        for (auto &edge : node[cur]) {
            int next = edge.first;
            int weight = edge.second;
            if (!visited[next]) {
                int newDist = dis[cur] + weight;
                if (newDist < dis[next]) {
                    dis[next] = newDist;
                }
            }
        }
    }
}


int main () {

    cin >> n >> m >> start;

    for (int i = 1; i <= n; i++) {
        visited[i] = false;
        dis[i] = INF;
    }

    for (int i = 0; i < m; i++) {
        int a, b, w;
        cin >> a >> b >> w;
        node[a].push_back({b, w});
    }

    // 다익스트라 진행
    dijkstra();

    // 각 노드에 도달하기 위해 필요한 weight 출력 (weight 지정 안되면 INF 출력)
    for (int i = 1; i <= n; i++) {
        if (dis[i] == INF) {
            cout << "INF" << '\n';
        } else {
            cout << dis[i] << '\n';
        }
    }

    return 0;
}

// https://codejin.tistory.com/193