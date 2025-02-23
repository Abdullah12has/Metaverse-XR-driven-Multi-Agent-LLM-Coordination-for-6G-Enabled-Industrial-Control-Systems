# **Project Validation and Troubleshooting Guide**

## **1. Component 3: Async Communication (Kafka)**
### **Validation Steps:**
#### **Check Kafka Topics:**
```bash
kubectl exec -it kafka-68599c8986-8gcdj -- /opt/bitnami/kafka/bin/kafka-topics.sh \
  --list --bootstrap-server localhost:9092
```
✅ Should show `sensor_topic`

#### **Verify Kafka Messages:**
```bash
kubectl exec -it kafka-68599c8986-8gcdj -- /opt/bitnami/kafka/bin/kafka-console-consumer.sh \
  --topic sensor_topic --from-beginning --bootstrap-server localhost:9092
```
✅ Should show continuous JSON sensor data

#### **Check Server Kafka Logs:**
```bash
kubectl logs deployment/server | grep "Kafka received"
```
✅ Should show received sensor data

---

## **2. Component 4: Leader Election & Consensus (Raft)**
### **Validation Steps:**
#### **Confirm Leader Election:**
```bash
kubectl logs deployment/server | grep "Elected leader"
```
✅ Example Output: `Elected leader at 10.244.0.5:50051 (Term 1)`

#### **Test Leader Redirection:**
```bash
kubectl logs deployment/client | grep "Redirecting"
```
✅ Clients should redirect to valid leader IPs

#### **Kill Leader Pod:**
```bash
kubectl delete pod <leader-pod-name>
```
✅ New leader should be elected automatically (check server logs)

---

## **3. Component 5: Auto-Scaling (Kubernetes HPA)**
### **Validation Steps:**
#### **Check HPA Status:**
```bash
kubectl get hpa
```
✅ Verify CPU utilization triggers scaling

#### **Generate Load:**
```bash
kubectl exec -it client-56ccb5499d-db4nn -- python3 scripts/load_test.py
```
✅ Load should trigger scaling

#### **Observe Scaling:**
```bash
watch kubectl get pods
```
✅ Server pods should scale up when load increases

---

## **4. Component 6: Resource Management**
### **Validation Steps:**
#### **Verify Resource Limits:**
```bash
kubectl describe pod server-74dfddc7c8-c2qlv | grep -A 5 "Resources"
```
✅ Should show CPU/memory limits from `server-deployment.yaml`

#### **Check Priority Classes:**
```bash
kubectl get priorityclass
```
✅ Verify `high-priority` class exists

---

## **5. Non-Functional Requirements**
### **Validation Steps:**
#### **Metrics Endpoint:**
```bash
kubectl port-forward deployment/server 8000:8000
```
✅ Open [http://localhost:8000/metrics](http://localhost:8000/metrics) - should show `grpc_requests_total`

#### **Latency Test:**
```bash
kubectl logs deployment/client | grep "Optimized settings received"
```
✅ Check for consistent response times (<1s)

#### **Chaos Test:**
```bash
kubectl delete pod kafka-68599c8986-8gcdj  # Kill Kafka
```
✅ Clients should buffer messages until Kafka recovers

---

## **6. Final Deliverables Checklist**
| **Requirement**            | **Validation Method**                                     |
|----------------------------|-----------------------------------------------------------|
| **System Design Diagram**  | Confirm diagram exists in project docs                   |
| **Demo Video (1-2 mins)**  | Verify video shows all components working                |
| **Performance Metrics**    | Check Prometheus/Grafana dashboard                        |
| **Auto-Scaling Proof**     | `kubectl describe hpa` + load test screenshots           |
| **Code Quality**           | All pods show 0 restarts in `kubectl get pods`            |

---

## **Troubleshooting Tips**
### **No Leader Elected:**
- Check Raft logic in `raft.py`
- Ensure `POD_IP` is injected in server pods:
  ```bash
  kubectl describe pod <server-pod-name>
  ```

### **Kafka Connection Issues:**
- Verify Kafka service DNS name:
  ```bash
  kubectl get svc kafka
  ```
- Check Zookeeper logs:
  ```bash
  kubectl logs zookeeper-5968d887b9-crpbx
  ```

### **gRPC Errors:**
- Verify `server-service:50051` exists:
  ```bash
  kubectl get svc server-service
  ```

---

✅ If all these checks pass, your system is **100% operational** and meets all project requirements! 🚀

