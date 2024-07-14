use std::{fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{make_array, Array, PrimitiveArray, RecordBatch, StringArray},
        datatypes::{DataType, Field, Float64Type, Schema, SchemaRef, UInt64Type},
        util::pretty::print_batches,
    },
    catalog::schema,
    datasource::{provider, TableProvider, TableType},
    error::DataFusionError,
    execution::{context::SessionState, SendableRecordBatchStream, TaskContext},
    logical_expr::Expr,
    parquet::file::properties,
    physical_expr::EquivalenceProperties,
    physical_plan::{
        stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionMode,
        ExecutionPlan, Partitioning, PlanProperties,
    },
    prelude::SessionContext,
};
use futures::{stream, Stream};

#[derive(Debug, Clone)]
struct MyEdges {
    id: PrimitiveArray<UInt64Type>,
    src: PrimitiveArray<UInt64Type>,
    dst: PrimitiveArray<UInt64Type>,
    weight: PrimitiveArray<Float64Type>,
}

#[async_trait]
impl TableProvider for MyEdges {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("src", DataType::UInt64, false),
            Field::new("dst", DataType::UInt64, false),
            Field::new("weight", DataType::Float64, false),
        ]))
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &SessionState,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let columns: Vec<Arc<dyn Array>> = vec![
            Arc::new(self.id.clone()),
            Arc::new(self.src.clone()),
            Arc::new(self.dst.clone()),
            Arc::new(self.weight.clone()),
        ];

        let columns = projection
            .map(|cols| {
                cols.iter()
                    .map(|&i| {
                        let col = make_array(columns[i].to_data());
                        col
                    })
                    .collect()
            })
            .unwrap_or(columns);

        let eq_properties = EquivalenceProperties::new(self.schema());

        let cores = std::thread::available_parallelism().unwrap().get();
        let partitioning = Partitioning::UnknownPartitioning(self.id.len().min(cores));
        let execution_mode = ExecutionMode::Bounded;
        let plan_properties = PlanProperties::new(eq_properties, partitioning, execution_mode);

        Ok(Arc::new(MyExecutionPlan {
            columns,
            plan_properties,
        }))
    }
}

struct MyNodes {
    id: PrimitiveArray<UInt64Type>,
    name: StringArray,
}

#[async_trait]
impl TableProvider for MyNodes {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
        ]))
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &SessionState,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let columns: Vec<Arc<dyn Array>> =
            vec![Arc::new(self.id.clone()), Arc::new(self.name.clone())];

        let columns = projection
            .map(|cols| {
                cols.iter()
                    .map(|&i| {
                        let col = make_array(columns[i].to_data());
                        col
                    })
                    .collect()
            })
            .unwrap_or(columns);

        let eq_properties = EquivalenceProperties::new(self.schema());

        let cores = std::thread::available_parallelism().unwrap().get();
        let partitioning = Partitioning::UnknownPartitioning(self.id.len().min(cores));
        let execution_mode = ExecutionMode::Bounded;
        let plan_properties = PlanProperties::new(eq_properties, partitioning, execution_mode);

        Ok(Arc::new(MyExecutionPlan {
            columns,
            plan_properties,
        }))
    }
}

#[derive(Debug, Clone)]
struct MyExecutionPlan {
    columns: Vec<Arc<dyn Array>>,
    plan_properties: PlanProperties,
}

impl DisplayAs for MyExecutionPlan {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MyExecutionPlan")
    }
}

#[async_trait]
impl ExecutionPlan for MyExecutionPlan {
    fn name(&self) -> &str {
        "MyExecutionPlan"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    /// Returns a new `ExecutionPlan` where all existing children were replaced
    /// by the `children`, in order
    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream, DataFusionError> {
        let partition_count = self.plan_properties.partitioning.partition_count();
        let stream_columns = stream_columns(
            self.columns.clone(),
            partition,
            partition_count,
            self.schema(),
        )?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream_columns,
        )))
    }
}

fn stream_columns(
    columns: Vec<Arc<dyn Array>>,
    partition: usize,
    partition_count: usize,
    schema: SchemaRef,
) -> Result<impl Stream<Item = Result<RecordBatch, DataFusionError>>, DataFusionError> {
    let len = columns[0].len();
    let part_size = len / partition_count;

    let batch = columns
        .into_iter()
        .filter_map(|col| {
            let start = part_size * partition;
            let end = start + part_size;
            let end = end.min(len);

            (start < len).then(|| {
                let col = col.slice(start, end - start);
                Ok::<_, DataFusionError>(col)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let rb = RecordBatch::try_new(schema, batch)?;

    Ok(stream::once(async move { Ok(rb) }))
}

#[tokio::main]
async fn main() {
    let session = SessionContext::new();

    let edges = MyEdges {
        id: PrimitiveArray::from(vec![0, 1, 2, 3, 4]),
        src: PrimitiveArray::from(vec![1, 2, 3, 4, 5]),
        dst: PrimitiveArray::from(vec![2, 3, 4, 5, 6]),
        weight: PrimitiveArray::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    };
    session
        .register_table("edges", Arc::new(edges))
        .expect("register edges");

    let nodes = MyNodes {
        id: PrimitiveArray::from(vec![1, 2, 3, 4, 5, 6]),
        name: StringArray::from(vec!["a", "b", "c", "d", "e", "f"]),
    };

    session
        .register_table("nodes", Arc::new(nodes))
        .expect("register nodes");

    let res = session
        .sql("SELECT * FROM edges")
        .await
        .expect("execute query");
    let res = res.collect().await.expect("collect results");

    print_batches(&res).expect("print edges");

    let res = session
        .sql("SELECT * FROM nodes")
        .await
        .expect("execute query");
    let res = res.collect().await.expect("collect results");

    print_batches(&res).expect("print nodes");

    let sql = "SELECT nodes.name FROM edges JOIN nodes ON edges.src = nodes.id";

    let res = session.sql(sql).await.expect("execute query");
    let res = res.collect().await.expect("collect results");

    print_batches(&res).expect("print batches");

    let sql = "WITH \
    e1 AS (SELECT * FROM edges), \
    a AS (SELECT * FROM nodes) \
    SELECT a.name \
    FROM e1 JOIN a ON e1.src = a.id";

    let res = session.sql(sql).await.expect("execute query");
    let res = res.collect().await.expect("collect results");

    print_batches(&res).expect("print batches");

    let sql = "WITH \
    e1 AS (SELECT * FROM edges), \
    e2 AS (SELECT * FROM edges), \
    a AS (SELECT * FROM nodes), \
    b AS (SELECT * FROM nodes), \
    c AS (SELECT * FROM nodes) 
    SELECT a.name, b.name, c.name \
    FROM e1 JOIN a ON e1.src = a.id \
    JOIN b ON e1.dst = b.id \
    JOIN e2 ON b.id = e2.src \
    JOIN c ON e2.dst = c.id \
    WHERE e1.id <> e2.id";

    let res = session.sql(sql).await.expect("execute query");
    let res = res.collect().await.expect("collect results");

    print_batches(&res).expect("print batches");
}
